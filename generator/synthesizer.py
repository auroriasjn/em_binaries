import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import uniform, norm

from typing import Tuple
from utils import safe_loc
from isochrones import get_ichrone
from isochrones.priors import GaussianPrior, SalpeterPrior, DistancePrior, FlatPrior
from isochrones.populations import StarFormationHistory, StarPopulation

class IsochroneSynthesizer:
    def __init__(
        self,
        mass_bounds: Tuple[float] = (0.1, 15.0),
        age: float = 1.1e8,
        feh: float = 0.0, 
        AV: float = 0.12,
        distance: float = 135,
        fB: float = 0.5
    ):
        self.mass_bounds = mass_bounds
        self.age = age
        self.feh = feh
        self.AV = AV
        self.distance = distance
        self.fB = fB

        self.mist = get_ichrone("mist", bands=["G", "BP", "RP"])
        self.mist.model_grid.df_loc = safe_loc

        self._init_fixed_cluster()

    def _init_fixed_cluster(self):
        self.imf = SalpeterPrior(bounds=self.mass_bounds)

        # Fixed priors for a single-aged cluster
        self.sfh = StarFormationHistory(dist=norm(loc=np.log10(self.age), scale=0.02))
        self.feh_prior = GaussianPrior(self.feh, 0.02)
        self.distance_prior = DistancePrior(max_distance=self.distance)
        self.AV_prior = GaussianPrior(self.AV, 0.02)

        self.pop = StarPopulation(
            self.mist,
            imf=self.imf,
            fB=self.fB,
            sfh=self.sfh,
            feh=self.feh_prior,
            distance=self.distance_prior,
            AV=self.AV_prior
        )

    def generate(self, n: int=1000):
        data = self.pop.generate(n)
        data['phot_bp_rp'] = data['BP_mag'] - data['RP_mag']

        return data
    
    def plot_hr_diagram(self, n: int=1000, show: bool=True):
        data = self.generate(n)

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.scatter(
            data['phot_bp_rp'],
            data['G_mag'],
            s=1,
            c='blue',
            alpha=0.5
        )
        ax.set_xlabel('BP - RP Color Index')
        ax.set_ylabel('G Mean Magnitude')
        ax.invert_yaxis()
        ax.set_title('CMD Diagram (Generated)')

        if show:
            plt.show()
            
        return fig, ax