import numpy as np
import matplotlib.pyplot as plt
import logging

from tqdm import tqdm
from typing import Tuple
from functools import lru_cache
from scipy.spatial import cKDTree

from utils import distance_modulus, flux_to_mag
from analyzer import MISTFitter


class BinaryMixtureFitter(MISTFitter):
    """
    Mixture model:
        p(x | θ) = π_single * L_single + π_binary * L_binary + π_field * L_field
    Where likelihoods are computed in CMD space using KD-tree matching.
    """
    def __init__(self, data, fB=0.2, mass_ratio=1.0,
                 q_range=(0.1, 1.0), field_weight=0.0,
                 use_field: bool = True, **kwargs):

        super().__init__(data, **kwargs)

        self.use_field = use_field

        # mixture weights
        if self.use_field:
            self.pi_single = 1.0 - fB - field_weight
            self.pi_binary = fB
            self.pi_field  = field_weight
            self.mixture_weights = np.array(
                [self.pi_single, self.pi_binary, self.pi_field]
            )
        else:
            self.pi_single = 1.0 - fB
            self.pi_binary = fB
            self.mixture_weights = np.array(
                [self.pi_single, self.pi_binary]
            )

        # binary properties
        self.mass_ratio = mass_ratio  # q
        self.qmin, self.qmax = q_range

        self._init_unc()

        # use best-fit single-star cluster parameters:
        self.theta = None

    def _init_unc(self):
        # observational uncertainties from MISTFitter
        if "G_mag_unc" in self.data:
            self.sM = np.nan_to_num(self.data["G_mag_unc"].to_numpy(), nan=self._sG_med)
        else:
            self.sM = np.full(len(self.data), self._sG_med)

        # BP and RP uncertainties (each with fallback)
        if "BP_mag_unc" in self.data:
            sBP = np.nan_to_num(self.data["BP_mag_unc"].to_numpy(), nan=self._sBP_med)
        else:
            sBP = np.full(len(self.data), self._sBP_med)

        if "RP_mag_unc" in self.data:
            sRP = np.nan_to_num(self.data["RP_mag_unc"].to_numpy(), nan=self._sRP_med)
        else:
            sRP = np.full(len(self.data), self._sRP_med)

        # Color uncertainty
        self.sC = np.sqrt(sBP**2 + sRP**2)

        # clamp to safe values
        self.sC = np.maximum(self.sC, 0.01)
        self.sM = np.maximum(self.sM, 0.01)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def get_binary_fraction(self):
        return self.mixture_weights[1]

    def get_mixture_weights(self):
        return self.mixture_weights

    # ---------------------------------------------------------
    # Binary isochrone (correct implementation using MIST EEP interpolation)
    # ---------------------------------------------------------
    @lru_cache(maxsize=32)
    def _compute_binary_iso(self, age, feh, distance, AV, q):
        logage = np.log10(age)

        # --- Primary: base isochrone at 10 pc, AV=0 ---
        self.mist.param_index_order = [1, 2, 0, 3, 4]
        iso = self.mist.isochrone(logage, feh=feh, distance=10.0, AV=0.0)

        mass1 = iso["mass"].to_numpy()
        G1  = iso["G_mag"].to_numpy()
        BP1 = iso["BP_mag"].to_numpy()
        RP1 = iso["RP_mag"].to_numpy()

        # --- Secondary: same age/feh, 10 pc, AV=0 ---
        mass2 = q * mass1
        try:
            eep2 = self.mist.get_eep(mass2, logage, feh, accurate=True)
        except ValueError as e:
            logging.error(f"Skipping binary computation: {e}")
            return np.array([]), np.array([]), np.array([])

        eep_min = self.mist.model_grid.interp.index_columns[2].min()
        eep_max = self.mist.model_grid.interp.index_columns[2].max()
        eep2 = np.clip(eep2, eep_min, eep_max)

        self.mist.param_index_order = [0, 1, 2, 3, 4]
        G2_list, BP2_list, RP2_list = [], [], []
        for e in eep2:
            try:
                _, _, _, g  = self.mist.interp_mag([logage, feh, e, 10.0, 0.0], ["G"])
                _, _, _, bp = self.mist.interp_mag([logage, feh, e, 10.0, 0.0], ["BP"])
                _, _, _, rp = self.mist.interp_mag([logage, feh, e, 10.0, 0.0], ["RP"])
                G2_list.append(g[0])
                BP2_list.append(bp[0])
                RP2_list.append(rp[0])
            except Exception as err:
                logging.info(f"interp_mag failed at EEP={e:.1f}: {err}")
                G2_list.append(np.nan)
                BP2_list.append(np.nan)
                RP2_list.append(np.nan)

        G2  = np.array(G2_list)
        BP2 = np.array(BP2_list)
        RP2 = np.array(RP2_list)

        # Combine fluxes at 10 pc, AV=0
        def combine(m1, m2):
            f1 = 10**(-0.4 * m1)
            f2 = 10**(-0.4 * m2)
            return -2.5 * np.log10(f1 + f2)

        G_bin_0  = combine(G1,  G2)
        BP_bin_0 = combine(BP1, BP2)
        RP_bin_0 = combine(RP1, RP2)

        # Apply distance + extinction exactly *once*
        DM = distance_modulus(distance)
        G0, BP0, RP0, dG, dBP, dRP = self._cache_iso_base(logage, feh)

        G_bin  = G_bin_0  + DM + dG  * AV
        BP_bin = BP_bin_0 + DM + dBP * AV
        RP_bin = RP_bin_0 + DM + dRP * AV

        return G_bin, BP_bin, RP_bin


    # ---------------------------------------------------------
    # Likelihoods
    # ---------------------------------------------------------
    def _binary_likelihood(self, theta):
        age, feh, distance, AV, dM, dC = theta
        isoG, isoBP, isoRP = self._compute_binary_iso(age, feh, distance, AV, q=self.mass_ratio)

        # If computation failed, return zeros of correct shape
        if isoG.size == 0:
            logging.warning("Binary isochrone returned empty array; using uniform low likelihood.")
            return np.full(len(self.data), -np.inf)

        iso_color = isoBP - isoRP
        iso_mag   = isoG

        obs_color = (self.BP - self.RP) - dC
        obs_mag   = self.G - dM

        tree = cKDTree(np.column_stack([iso_color, iso_mag]))
        idx = tree.query(np.column_stack([obs_color, obs_mag]), k=1)[1]

        rC = obs_color - iso_color[idx]
        rM = obs_mag   - iso_mag[idx]

        # --- Intrinsic scatter added here ---
        sigmaC_int = 0.15  # intrinsic color scatter (mag)
        sigmaM_int = 0.30  # intrinsic mag scatter (mag)
        sC_eff = np.sqrt(self.sC**2 + sigmaC_int**2)
        sM_eff = np.sqrt(self.sM**2 + sigmaM_int**2)

        # Return per-star log-likelihood array
        lnL = -0.5 * ((rC / sC_eff)**2 + (rM / sM_eff)**2)
        return lnL

    def _single_likelihood(self, theta):
        # Compute isochrone residuals per star
        rC, rM, sC, sM = self._compute_residuals(theta)

        # asymmetric weights (tune lambdas)
        λC = 1.0
        λM = 1.0
        wC = 1.0 + λC * (rC < 0)
        wM = 1.0 + λM * (rM < 0)

        # --- Intrinsic scatter added here ---
        sigmaC_int = 0.15
        sigmaM_int = 0.30
        sC_eff = np.sqrt(sC**2 + sigmaC_int**2)
        sM_eff = np.sqrt(sM**2 + sigmaM_int**2)

        chi2 = wC * (rC / sC_eff)**2 + wM * (rM / sM_eff)**2

        # Per-star log-likelihoods (no sum!)
        lnL = -0.5 * chi2
        return lnL

    def _field_likelihood(self, theta):
        """
        Uniform PDF over CMD bounding box.
        """
        cmin, cmax = np.min(self.BP - self.RP), np.max(self.BP - self.RP)
        mmin, mmax = np.min(self.G),          np.max(self.G)
        area = (cmax - cmin) * (mmax - mmin)

        # uniform log-likelihood
        return -np.log(area) * np.ones(len(self.data))

    # ---------------------------------------------------------
    # EM Algorithm
    # ---------------------------------------------------------
    def E_step(self, theta, tau: float = 1.5):
        # per-star log-likelihoods
        ls = self._single_likelihood(theta)
        lb = self._binary_likelihood(theta)

        # --- Optional median normalization for scale parity ---
        ls -= np.nanmedian(ls)
        lb -= np.nanmedian(lb)

        log_pi = np.log(self.mixture_weights + 1e-300)

        if self.use_field:
            lf = self._field_likelihood(theta)
            lf -= np.nanmedian(lf)
            log_post = np.column_stack([
                log_pi[0] + ls,
                log_pi[1] + lb,
                log_pi[2] + lf
            ]) / max(tau, 1.0)
        else:
            log_post = np.column_stack([
                log_pi[0] + ls,
                log_pi[1] + lb,
            ]) / max(tau, 1.0)

        log_post -= np.max(log_post, axis=1, keepdims=True)
        R = np.exp(log_post)
        R /= np.sum(R, axis=1, keepdims=True)
        R[~np.isfinite(R)] = 0.0

        return R

    def M_step(self, R):
        logging.debug("=== M-step start ===")

        Nk = R.sum(axis=0)
        Nk[~np.isfinite(Nk)] = 0.0

        if self.use_field:
            alpha = np.array([6.0, 3.0, 1.5])
        else:
            alpha = np.array([6.0, 3.0])  # only 2 components

        Nk += (alpha - 1.0)

        self.mixture_weights = Nk / Nk.sum()

        # numeric safety
        eps = 1e-6
        self.mixture_weights = np.clip(self.mixture_weights, eps, 1 - eps)
        self.mixture_weights /= self.mixture_weights.sum()

        logging.info(f"Updated mixture weights: {self.mixture_weights}")
        logging.debug("=== M-step complete ===")

    # ---------------------------------------------------------
    # Fit mixture model, cluster params frozen at θ
    # ---------------------------------------------------------
    def fit(self, theta, n_iterations=40):
        """
        theta = best-fit cluster parameters from MISTFitter (6-vector)
        """
        self.theta = theta
        logging.info("Starting EM fit")
        logging.info(f"Initial weights: {self.mixture_weights}")

        for i in tqdm(range(n_iterations)):
            logging.debug(f"\n=== EM Iteration {i+1}/{n_iterations} ===")
            R = self.E_step(theta)
            self.M_step(R)

            # sanity check: NaN weights
            if not np.all(np.isfinite(self.mixture_weights)):
                logging.warning(f"Iteration {i+1}: mixture weights became NaN!")
                break

        logging.info(f"Final mixture weights: {self.mixture_weights}")
        return self
    
    def get_probability(self, source_id: str):
        if self.theta is None:
            raise RuntimeError("Error: fit() must be run first.")
        
        # Compute responsibilities
        R = self.E_step(self.theta)   # shape = (N, 3)
        
        # Find the index of the star in your dataset
        df = self.data
        idx = df.index[df['source_id'].astype(str) == str(source_id)]
        
        if len(idx) == 0:
            raise ValueError(f"Source ID {source_id} not found in dataset.")
        
        i = idx[0]
        res = {
            "single": float(R[i, 0]),
            "binary": float(R[i, 1]),
        }
        if self.use_field:
            res["field"] = float(R[i, 2])
        
        return res

    def plot(self):
        if self.theta is None:
            raise RuntimeError("Error: fit() must be run first.")
    
        R = self.E_step(self.theta)
        if self.use_field:
            labels = ["Single", "Binary", "Field"]
            colors = [R[:,0], R[:,1], R[:,2]]

            fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharex=True, sharey=True)
        else:
            labels = ["Single", "Binary"]
            colors = [R[:,0], R[:,1]]

            fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)

        for ax, c, lbl in zip(axes, colors, labels):
            sc = ax.scatter(self.BP - self.RP, self.G,
                            c=c, cmap="plasma", s=10)
            ax.set_title(lbl)
            plt.colorbar(sc, ax=ax, label=f"P({lbl})")
            ax.set_xlabel("(BP − RP)")
            ax.set_ylabel("G magnitude")
        
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(max(ymin, ymax), min(ymin, ymax))
        
        plt.tight_layout()
        plt.show()

        return fig, axes
