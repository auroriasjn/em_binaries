import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List

class GaiaAnalyzer:    
    def __init__(
        self,
        filename: str='gaia.csv'
    ):
        self.filename = filename
        self.data = pd.read_csv(self.filename)

        # Initialize colors
        self._init_params()

        # Init Uncertainties
        self._init_uncertainties()

    def _init_params(self):
        if 'phot_bp_rp' not in self.data.columns:
            self.data['phot_bp_rp'] = self.data['phot_bp_mean_mag'] - self.data['phot_rp_mean_mag']
        
        self.data = self.data.rename(columns={
            'teff_gspphot': 'teff',
            'logg_gspspec': 'logg',
            'distance_gspphot': 'distance',
            'phot_bp_mean_mag': 'BP_mag',
            'phot_rp_mean_mag': 'RP_mag',
            'phot_g_mean_mag': 'G_mag',
        })
        
    def _init_uncertainties(self):
        # Rename parallax
        self.data = self.data.rename(columns={
            'parallax_error': 'parallax_unc',
        })

        # Calculate color errors if not present
        for band in ('BP', 'RP', 'G'):
            flux_col = f'phot_{band.lower()}_mean_flux'
            flux_err_col = f'phot_{band.lower()}_mean_flux_error'
            err_col = f'{band}_mag_unc'

            if err_col not in self.data.columns and flux_col in self.data.columns and flux_err_col in self.data.columns:
                self.data[err_col] = (2.5 / np.log(10)) * (self.data[flux_err_col] / self.data[flux_col])

        # Calculate uncertainties via upper and lower
        if 'teff_gspphot_upper' in self.data.columns and 'teff_gspphot_lower' in self.data.columns:
            self.data['teff_unc'] = 0.5 * (self.data['teff_gspphot_upper'] - self.data['teff_gspphot_lower'])
        if 'logg_gspspec_upper' in self.data.columns and 'logg_gspspec_lower' in self.data.columns:
            self.data['logg_unc'] = 0.5 * (self.data['logg_gspspec_upper'] - self.data['logg_gspspec_lower'])
        if 'distance_gspphot_upper' in self.data.columns and 'distance_gspphot_lower' in self.data.columns:
            self.data['distance_unc'] = 0.5 * (self.data['distance_gspphot_upper'] - self.data['distance_gspphot_lower'])

    def get_data(self) -> pd.DataFrame:
        return self.data

    def plot_hr_diagram(self, show: bool=True):
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.scatter(
            self.data['phot_bp_rp'],
            self.data['G_mag'],
            s=1,
            c='blue',
            alpha=0.5
        )
        ax.set_xlabel('BP - RP Color Index')
        ax.set_ylabel('G Mean Magnitude')
        ax.invert_yaxis()
        ax.set_title('CMD Diagram')

        if show:
            plt.show()
            
        return fig, ax
        