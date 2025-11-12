import logging

import matplotlib.pyplot as plt
import numpy as np

from typing import List, Tuple, Any, Dict
from functools import lru_cache
from astropy.timeseries import LombScargle, BoxLeastSquares

class TESSAnalyzer:
    def __init__(
        self,
        lightcurves: Dict[str, Tuple[np.ndarray, np.ndarray, Any]]
    ):
        self.lightcurves = lightcurves
        self.results = dict()

    # --- GETTERS ---
    def extract_source_ids(self):
        # Placeholder method to extract source IDs
        return list(self.lightcurves.keys())

    def get_lightcurves(self):
        # Placeholder method to get data
        return self.lightcurves
    
    def get_lightcurve(self, source_id: str) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Retrieve a specific lightcurve by Gaia source ID."""
        if source_id not in self.lightcurves:
            logging.warning(f"Source ID {source_id} not found in lightcurves.")
            return None, None, None
        return self.lightcurves[source_id]

    # --- ANALYZERS ---
    @lru_cache(maxsize=32)
    def _prep(self, time, flux):
        m = np.isfinite(time) & np.isfinite(flux)
        t, f = time[m], flux[m]
        f = f / np.nanmedian(f) - 1.0
        return t, f

    # --- Lomb-Scargle ---
    def run_ls(self, time, flux, min_period=0.1, max_period=100):
        time, flux = self._prep(time, flux)

        freq, power = LombScargle(time, flux).autopower(
            minimum_frequency=1/max_period,
            maximum_frequency=1/min_period,
            samples_per_peak=5
        )
        period = 1 / freq[np.argmax(power)]
        return period, freq, power
    
    # --- BLS search (boxy eclipses) ---
    def run_bls(self, time, flux, min_period=0.1, max_period=100.0):
        t, f = self._prep(time, flux)
        periods = np.exp(np.linspace(np.log(min_period), np.log(max_period), 5000))

        # Duration grid (fraction of period, capped below min_period)
        max_frac = 0.2                     # 20 % of period
        durations = np.linspace(0.02, 0.1, 10) * periods.mean()
        
        # Safety: ensure all durations < all periods
        durations = durations[durations < max_period]

        bls = BoxLeastSquares(t, f)
        res = bls.power(periods, durations)

        i = np.argmax(res.power)
        return {
            "period": res.period[i],
            "duration": res.duration[i],
            "t0": res.transit_time[i],
            "power": res.power[i],
            "power_array": res.power,
            "period_grid": res.period
        }

    def estimate_period(self, time, flux):
        p_ls, _, _ = self.run_ls(time, flux)
        bls = self.run_bls(time, flux)
        p_bls = bls["period"]

        # crude heuristic: prefer BLS if strong boxiness
        if bls["power"] > np.median(bls["power_array"]) + 5*np.std(bls["power_array"]):
            return p_bls, "BLS"
        else:
            return p_ls, "LS"

    def phase_fold(self, time: np.ndarray, flux: np.ndarray, period: float, t0: float = None):
        """
        Phase-fold lightcurve given a period and reference epoch (t0).
        """
        if t0 is None:
            t0 = time[0]
        phase = ((time - t0) / period) % 1
        phase[phase > 0.5] -= 1.0   # shift range from [0,1) → [-0.5,0.5)
        sort_idx = np.argsort(phase)
        return phase[sort_idx], flux[sort_idx]

    def analyze_lightcurve(self, flux, time):
        """
        Full single-lightcurve analysis pipeline:
        - Estimate period
        - Phase-fold
        """
        try:
            period, _ = self.estimate_period(time, flux)
            phase, folded_flux = self.phase_fold(time, flux, period)
            result = {
                "period": period,
                "phase": phase,
                "folded_flux": folded_flux
            }
            return result
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            return {"status": "failed"}
    
    def batch_analyze(self) -> Dict[str, Dict[str, Any]]:
        """
        Apply analyze_lightcurve() to every entry in self.lightcurves.
        Returns a dictionary mapping source IDs to analysis results.
        """
        results = {}
        for source_id, (time, flux, lc_obj) in self.lightcurves.items():
            logging.info(f"Analyzing lightcurve for {source_id}")
            result = self.analyze_lightcurve(flux, time)
            results[source_id] = result
        self.results = results
        return results
    
    # --- PLOTTING ---
    def plot_lightcurve(self, time: np.ndarray, flux: np.ndarray, source_id: str = None):
        """Plot an individual lightcurve."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time, flux, lw=0.7, color="black")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Normalized Flux")
        title = f"TESS Lightcurve" + (f" - {source_id}" if source_id else "")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def batch_plot(self, nmax: int = 5):
        """
        Plot up to nmax lightcurves from the dictionary.
        Each lightcurve is plotted separately.
        """
        ids = list(self.lightcurves.keys())[:nmax]
        if not ids:
            logging.warning("No lightcurves available to plot.")
            return

        for source_id in ids:
            time, flux, _ = self.lightcurves[source_id]
            self.plot_lightcurve(time, flux, source_id)

        plt.show()

    def plot_phase_folded(self, phase: np.ndarray, flux: np.ndarray, source_id: str = None, period: float = None):
        """Plot a phase-folded lightcurve."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(phase, flux, s=6, color="black", alpha=0.6)
        ax.set_xlabel("Orbital Phase")
        ax.set_ylabel("Normalized Flux")
        title = "Phase-Folded Lightcurve"
        if source_id:
            title += f" - {source_id}"
        if period:
            title += f" (P={period:.4f} d)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def batch_plot_folded(self, nmax: int = 5):
        """
        Plot up to nmax phase-folded lightcurves after batch_analyze().
        """
        if not self.results:
            logging.warning("Run batch_analyze() first to compute periods.")
            return

        ids = list(self.results.keys())[:nmax]
        for source_id in ids:
            res = self.results[source_id]
            if res is None or "phase" not in res:
                continue
            self.plot_phase_folded(res["phase"], res["folded_flux"], source_id, res["period"])
        plt.show()