import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from functools import lru_cache
from scipy.spatial import cKDTree

from isochrones import get_ichrone
from utils import distance_modulus

class MISTFitter:
    def __init__(
        self,
        data: pd.DataFrame,
        age_range: Tuple[float, float] = (90e6, 160e6),
        feh_range: Tuple[float, float] = (-0.2, 0.2),
        AV_range: Tuple[float, float] = (0.0, 0.6),
        # Nuisance shift parameters
        dM_range: Tuple[float, float] = (-0.15, 0.15),
        dC_range: Tuple[float, float] = (-0.08, 0.08),
    ):
        """
        Fit Pleiades-like cluster using Gaia G/BP/RP.
        Expects columns: G_mag, BP_mag, RP_mag; optional *_mag_unc, distance, distance_unc.
        """
        self.data = data.reset_index(drop=True)
        self._prep_params()

        self.age_range = age_range
        self.feh_range = feh_range
        self.AV_range = AV_range
        self.dM_range = dM_range
        self.dC_range = dC_range

        # distance prior: use data if present, else Pleiades default window
        self.distance_range = self._compute_distance_range(default=(110.0, 170.0))

        self.mist = get_ichrone("mist", bands=["G", "BP", "RP"])

        # cache median uncertainty fallbacks
        self._sG_med  = np.nanmedian(self.data.get("G_mag_unc",  np.array([0.03])))
        self._sBP_med = np.nanmedian(self.data.get("BP_mag_unc", np.array([0.03])))
        self._sRP_med = np.nanmedian(self.data.get("RP_mag_unc", np.array([0.03])))

    def _compute_distance_range(self, default=(110.0, 170.0)):
        if "distance" not in self.data.columns:
            return default
        
        d = self.data["distance"].to_numpy()
        dmin, dmax = np.nanmin(d), np.nanmax(d)

        if "distance_unc" in self.data.columns:
            unc = np.nanmedian(self.data["distance_unc"])
            if np.isfinite(unc):
                dmin = max(dmin - 3 * unc, 10.0)
                dmax = dmax + 3 * unc
        
        # clip to a reasonable Pleiades envelope if crazy values exist
        if dmin >= dmax:
            return default
    
        return (dmin, dmax)
    
    def _prep_params(self):
        self.BP = self.data["BP_mag"].to_numpy()
        self.RP = self.data["RP_mag"].to_numpy()
        self.G  = self.data["G_mag"].to_numpy()

    # ----- caching: base isochrone at 10 pc, AV=0 and per-point extinction slopes -----
    @lru_cache(maxsize=128)
    def _cache_iso_base(self, logage: float, feh: float, dAV: float = 0.05):
        iso0 = self.mist.isochrone(logage, feh=feh, distance=10.0, AV=0.0)
        isoA = self.mist.isochrone(logage, feh=feh, distance=10.0, AV=dAV)

        G0  = iso0["G_mag"].to_numpy()
        BP0 = iso0["BP_mag"].to_numpy()
        RP0 = iso0["RP_mag"].to_numpy()

        dG  = (isoA["G_mag"].to_numpy()  - G0)  / dAV
        dBP = (isoA["BP_mag"].to_numpy() - BP0) / dAV
        dRP = (isoA["RP_mag"].to_numpy() - RP0) / dAV

        return G0, BP0, RP0, dG, dBP, dRP

    # ----- priors -----
    def ln_prior(self, theta):
        age, feh, distance, AV, dM, dC = theta

        if not (self.age_range[0]     < age      < self.age_range[1]):     return -np.inf
        if not (self.feh_range[0]     < feh      < self.feh_range[1]):     return -np.inf
        if not (self.distance_range[0] < distance < self.distance_range[1]): return -np.inf
        if not (self.AV_range[0]      < AV       < self.AV_range[1]):      return -np.inf
        if not (self.dM_range[0]      < dM       < self.dM_range[1]):      return -np.inf
        if not (self.dC_range[0]      < dC       < self.dC_range[1]):      return -np.inf

        return 0.0

    # ----- 2-D CMD likelihood -----
    def ln_likelihood(self, theta):
        age, feh, distance, AV, dM, dC = theta
        logage = np.log10(age)

        # base + per-point AV response
        G0, BP0, RP0, dG, dBP, dRP = self._cache_iso_base(logage, feh)
        DM = distance_modulus(distance)

        # apply distance + extinction
        iso_G  = G0  + DM + dG  * AV
        iso_BP = BP0 + DM + dBP * AV
        iso_RP = RP0 + DM + dRP * AV

        iso_color = iso_BP - iso_RP
        iso_mag   = iso_G

        # observations (apply tiny global shifts to absorb ZP/systematics)
        BP, RP, G = self.BP, self.RP, self.G
        color_obs = (BP - RP) - dC
        mag_obs   = G - dM

        # per-star uncertainties with safe fallbacks
        sG  = np.nan_to_num(self.data.get("G_mag_unc",  np.full_like(G,  self._sG_med )).to_numpy(),  nan=self._sG_med)
        sBP = np.nan_to_num(self.data.get("BP_mag_unc", np.full_like(BP, self._sBP_med)).to_numpy(),  nan=self._sBP_med)
        sRP = np.nan_to_num(self.data.get("RP_mag_unc", np.full_like(RP, self._sRP_med)).to_numpy(),  nan=self._sRP_med)
        sC  = np.sqrt(sBP**2 + sRP**2)
        sM  = sG

        # Clamping
        sC, sM = np.maximum(sC, 0.01), np.maximum(sM, 0.01)

        # KD-tree in standardized space (use medians to avoid rebuilding tree per-star)
        sC_med = np.median(sC)
        sM_med = np.median(sM)
        tree = cKDTree(np.column_stack([iso_color / sC_med, iso_mag / sM_med]))
        idx = tree.query(np.column_stack([color_obs / sC_med, mag_obs / sM_med]), k=1)[1]

        mC = iso_color[idx]
        mM = iso_mag[idx]

        # residuals
        rC = color_obs - mC     # negative => bluer than isochrone
        rM = mag_obs   - mM     # negative => brighter than isochrone

        # asymmetric weights (tune lambdas)
        λC = 1.7
        λM = 1.4
        wC = 1.0 + λC * (rC < 0)
        wM = 1.0 + λM * (rM < 0)

        chi2 = wC * (rC / sC)**2 + wM * (rM / sM)**2
        return -0.5 * np.nansum(chi2)

    def ln_posterior(self, theta):
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.ln_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll

    # ----- sampling -----
    def sample_cluster(self, n_walkers=40, n_burn=600, n_steps=1500, seed=None, progress=True):
        rng = np.random.default_rng(seed)
        ndim = 6  # (age, feh, distance, AV, dM, dC)
        p0 = np.array([
            rng.uniform(*self.age_range, size=n_walkers),
            rng.uniform(*self.feh_range, size=n_walkers),
            rng.uniform(*self.distance_range, size=n_walkers),
            rng.uniform(*self.AV_range, size=n_walkers),
            rng.uniform(*self.dM_range, size=n_walkers),
            rng.uniform(*self.dC_range, size=n_walkers),
        ]).T
        sampler = emcee.EnsembleSampler(n_walkers, ndim, self.ln_posterior)
        p0, _, _ = sampler.run_mcmc(p0, n_burn, progress=progress)
        
        sampler.reset()
        sampler.run_mcmc(p0, n_steps, progress=progress)
        return sampler

    # ----- viz -----
    def plot_best_fit_isochrone(self, theta, show: bool=True):
        age, feh, distance, AV, dM, dC = theta

        logage = np.log10(age)
        G0, BP0, RP0, dG, dBP, dRP = self._cache_iso_base(logage, feh)
        DM = distance_modulus(distance)
        
        iso_G  = G0  + DM + dG  * AV
        iso_BP = BP0 + DM + dBP * AV
        iso_RP = RP0 + DM + dRP * AV

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.scatter(
            self.data['phot_bp_rp'],
            self.data['G_mag'],
            s=1,
            c='blue',
            alpha=0.5
        )
        ax.plot(iso_BP - iso_RP + dC, iso_G + dM, color="red", lw=2, label="Best-fit MIST Isochrone")
        
        ax.set_xlabel('BP - RP Color Index')
        ax.set_ylabel('G Mean Magnitude')
        ax.invert_yaxis()
        ax.set_title('CMD Diagram')

        if show:
            plt.show()
            
        return fig, ax
