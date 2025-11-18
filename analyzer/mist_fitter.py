import emcee
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt

from typing import Tuple
from functools import lru_cache
from scipy.spatial import cKDTree

from isochrones import get_ichrone
from utils import distance_modulus, safe_loc

class MISTFitter:
    def __init__(
        self,
        data: pd.DataFrame,
        age_range: Tuple[float, float] = (90e6, 160e6),
        feh_range: Tuple[float, float] = (-0.4, 0.4),
        AV_range: Tuple[float, float] = (0.0, 0.6),
        # Nuisance shift parameters
        dM_range: Tuple[float, float] = (-0.5, 0.5),
        dC_range: Tuple[float, float] = (-0.3, 0.3),
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
        self.mist.model_grid.df_loc = safe_loc

        # cache median uncertainty fallbacks
        self._sG_med  = np.nanmedian(self.data.get("G_mag_unc",  np.array([0.03])))
        self._sBP_med = np.nanmedian(self.data.get("BP_mag_unc", np.array([0.03])))
        self._sRP_med = np.nanmedian(self.data.get("RP_mag_unc", np.array([0.03])))

        # Cache early.
        self.best_model = None

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
        self.mist.param_index_order = [1, 2, 0, 3, 4]

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

        if not (self.age_range[0] < age < self.age_range[1]):
            return -np.inf
        if not (self.feh_range[0] < feh < self.feh_range[1]): 
            return -np.inf
        if not (self.distance_range[0] < distance < self.distance_range[1]): 
            return -np.inf
        if not (self.AV_range[0] < AV < self.AV_range[1]):      
            return -np.inf
        if not (self.dM_range[0] < dM < self.dM_range[1]):      
            return -np.inf
        if not (self.dC_range[0] < dC < self.dC_range[1]):      
            return -np.inf

        return 0.0

    # ----- 2-D CMD likelihood -----
    def _compute_iso_colors(self, age, feh, distance, AV):
        logage = np.log10(age)

        # base + per-point AV response
        G0, BP0, RP0, dG, dBP, dRP = self._cache_iso_base(logage, feh)
        DM = distance_modulus(distance)

        # apply distance + extinction
        iso_G  = G0  + DM + dG  * AV
        iso_BP = BP0 + DM + dBP * AV
        iso_RP = RP0 + DM + dRP * AV

        return iso_G, iso_BP, iso_RP
    
    def _compute_residuals(self, theta):
        age, feh, distance, AV, dM, dC = theta
        iso_G, iso_BP, iso_RP = self._compute_iso_colors(age, feh, distance, AV)

        iso_color, iso_mag = iso_BP - iso_RP, iso_G

        BP, RP, G = self.BP, self.RP, self.G
        color_obs = (BP - RP) - dC
        mag_obs   = G - dM

        # per-star uncertainties with safe fallbacks
        sG_source  = self.data.get("G_mag_unc",  np.full_like(G,  self._sG_med))
        sBP_source = self.data.get("BP_mag_unc", np.full_like(BP, self._sBP_med))
        sRP_source = self.data.get("RP_mag_unc", np.full_like(RP, self._sRP_med))

        sG  = np.nan_to_num(np.asarray(sG_source),  nan=self._sG_med)
        sBP = np.nan_to_num(np.asarray(sBP_source), nan=self._sBP_med)
        sRP = np.nan_to_num(np.asarray(sRP_source), nan=self._sRP_med)
        
        sC  = np.sqrt(sBP**2 + sRP**2)
        sM  = sG

        # Clamping
        sC, sM = np.maximum(sC, 0.01), np.maximum(sM, 0.01)

        # KD-tree in standardized space (use medians to avoid rebuilding tree per-star)
        sC_med = np.median(sC)
        sM_med = np.median(sM)

        iso_scaled = np.column_stack([iso_color / sC_med, iso_mag / sM_med])
        obs_scaled = np.column_stack([color_obs / sC_med, mag_obs / sM_med])

        tree = cKDTree(iso_scaled)
        dist, idx = tree.query(obs_scaled, k=1)

        mC = iso_color[idx]
        mM = iso_mag[idx]

        # residuals
        rC = color_obs - mC     # negative => bluer than isochrone
        rM = mag_obs   - mM     # negative => brighter than isochrone

        return rC, rM, sC, sM

    def ln_likelihood(self, theta):
        # Compute isochrone points
        rC, rM, sC, sM = self._compute_residuals(theta)

        # asymmetric weights (tune lambdas)
        λC = 1.0
        λM = 1.0
        wC = 1.0 + λC * (rC < 0)
        wM = 1.0 + λM * (rM < 0)

        chi2 = wC * (rC / sC)**2 + wM * (rM / sM)**2
        return -0.5 * np.nanmean(chi2)

    def ln_posterior(self, theta):
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.ln_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll
    
    def bic(self):
        if self.best_model is None:
            raise RuntimeError("Error: run sample_cluster() before computing BIC.")

        k = 6  # number of free parameters
        N = len(self.data)  # number of stars

        logL = self.ln_likelihood(self.best_model)

        return k * np.log(N) - 2 * logL

    # ----- sampling -----
    def sample_cluster(self, n_walkers=40, n_burn=600, n_steps=1500, seed=None, progress=True):
        # Adapted from Lab 3

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

        samples = sampler.get_chain(flat=True)

        # Save median model
        median_params = np.median(samples, axis=0)
        self.median_model = median_params

        # Save best model
        chi2_vals = np.array([self.ln_likelihood(theta) for theta in samples])
        self.chi2_vals = chi2_vals
        self.samples   = samples

        idx_best = np.argmax(chi2_vals)
        self.best_model = samples[idx_best]
        self.best_chi2  = chi2_vals[idx_best]

        return sampler
    
    def get_median_model(self):
        if self.median_model is None:
            raise RuntimeError("Error: must run sample_cluster() before getting median model.")
        return self.median_model

    def get_best_model(self):
        if self.best_model is None:
            raise RuntimeError("Error: must run sample_cluster() before getting best model.")
        return self.best_model
    
    def get_samples(self):
        if self.samples is None:
            raise RuntimeError("Error: must run sample_cluster() before getting samples.")
        return self.samples
    
    def get_data(self):
        return self.data

    def get_good_models(self, chi2_cutoff, max_models: int=200, seed: int=42):
        chi2 = np.asarray(self.chi2_vals)
        samples = self.samples

        ok = np.isfinite(chi2) & (chi2 <= chi2_cutoff)
        good = samples[ok]

        if len(good) > max_models:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(good), size=max_models, replace=False)
            good = good[idx]

        return good

    # --- Visualizations ---
    def plot_isochrone(self, theta, show: bool=True):
        age, feh, distance, AV, dM, dC = theta
        iso_G, iso_BP, iso_RP = self._compute_iso_colors(age, feh, distance, AV)

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
    
    def plot_good_isochrones(
        self,
        chi2_cutoff,
        max_curves=50,
        seed=42,
        dark=True,
        show=True
    ):
        """
        Overplot all isochrones with chi2 <= chi2_cutoff.
        """
        good_models = self.get_good_models(chi2_cutoff, max_models=max_curves, seed=seed)

        fig, ax = plt.subplots(figsize=(8,10))

        # scatter data
        ax.scatter(
            self.data['phot_bp_rp'], 
            self.data['G_mag'], 
            s=2, c=('blue' if not dark else 'yellow'), alpha=0.4
        )

        # plot each good model
        for theta in good_models:
            age, feh, distance, AV, dM, dC = theta
            iso_G, iso_BP, iso_RP = self._compute_iso_colors(age, feh, distance, AV)
            
            ax.plot(
                iso_BP - iso_RP + dC,
                iso_G + dM,
                color=('red' if not dark else 'orange'), alpha=(0.40 if dark else 0.05), lw=1
            )

        # highlight best model
        age, feh, distance, AV, dM, dC = self.best_model
        iso_G, iso_BP, iso_RP = self._compute_iso_colors(age, feh, distance, AV)

        ax.plot(
            iso_BP - iso_RP + dC,
            iso_G + dM,
            color=("black" if not dark else 'white'), lw=2.5, label=f"Best Model ($-\chi^2/2={self.best_chi2:.1f}$)"
        )

        ax.set_xlabel("BP - RP")
        ax.set_ylabel("G")
        ax.invert_yaxis()
        ax.legend()

        if show:
            plt.show()

        return fig, ax
    
    def plot_residuals(self, theta, show: bool=True, dark: bool=True):
        rC, rM, _, _ = self._compute_residuals(theta)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(
            self.data['phot_bp_rp'] - (theta[5]),  # dC
            rC,
            s=2,
            c='green',
            alpha=0.5
        )
        ax1.axhline(0, color=('black' if not dark else 'white'), ls='--')
        ax1.set_xlabel('BP - RP Color Index (corrected)')
        ax1.set_ylabel('Color Residual (Observed - Model)')
        ax1.set_title('Color Residuals')

        ax2.scatter(
            self.data['G_mag'] - (theta[4]),  # dM
            rM,
            s=2,
            c='purple',
            alpha=0.5
        )
        ax2.axhline(0, color=('black' if not dark else 'white'), ls='--')
        ax2.set_xlabel('G Mean Magnitude (corrected)')
        ax2.set_ylabel('Magnitude Residual (Observed - Model)')
        ax2.set_title('Magnitude Residuals')

        if show:
            plt.show()

        return fig, (ax1, ax2)

    def plot_corner(self, sampler, discard: int = 200, thin: int = 100, flat: bool = True):
        if self.best_model is None:
            raise RuntimeError("Error: must run sample_cluster() before plotting corner.")

        # Get flattened MCMC samples
        samples = sampler.get_chain(discard=discard, thin=thin, flat=flat)

        # Parameter labels and units
        labels = [
            r"$\mathrm{Age\ [yr]}$",
            r"$\mathrm{[Fe/H]}$",
            r"$\mathrm{d\ [pc]}$",
            r"$A_V$",
            r"$\Delta M$",
            r"$\Delta C$"
        ]

        # Generate the corner plot
        figure = corner.corner(
            samples,
            labels=labels,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            title_fmt=".3f",
            title_kwargs={"fontsize": 12},
            truths=self.best_model
        )

        plt.show()
        return figure