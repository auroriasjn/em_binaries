import matplotlib.pyplot as plt

from scipy.stats import uniform, norm

from typing import Tuple
from isochrones import get_ichrone
from isochrones.priors import GaussianPrior, SalpeterPrior, DistancePrior, FlatPrior
from isochrones.populations import StarFormationHistory, StarPopulation

class IsochroneSynthesizer:
    def __init__(
        self,
        mass_bounds: Tuple[float] = (1, 3),
        age_bounds: Tuple[float, float] = (0.7, 1), # in Gyr
        feh_range: Tuple[float, float] = (-0.4, 0.4),
        AV_range: Tuple[float, float] = (0.0, 0.6),
        max_distance: int = 3000,
        fB: float =0.5,
        gamma: float =0.3
    ):
        # Parameter initialization
        self.mass_bounds = mass_bounds
        self.age_bounds = age_bounds
        self.feh_range = feh_range
        self.AV_range = AV_range

        self.fB = fB
        self.gamma = gamma
        self.max_distance = max_distance

        # MIST and prior initialization
        self.mist = get_ichrone("mist", bands=["G", "BP", "RP"])

        self._init_priors()

    def _init_priors(self):
        self.imf = SalpeterPrior(bounds=self.mass_bounds)
        self.sfh = StarFormationHistory(dist=uniform(*self.age_bounds))
        self.feh = GaussianPrior(*self.feh_range)
        self.distance = DistancePrior(max_distance=self.max_distance)
        self.AV = FlatPrior(bounds=self.AV_range)

        self.pop = StarPopulation(
            self.mist,
            imf=self.imf,
            fB=self.fB,
            sfh=self.sfh,
            feh=self.feh,
            distance=self.distance,
            AV=self.AV
        )

    def generate(self, n: int=1000):
        return self.pop.generate(n)
    
    def plot_hr_diagram(self, n: int=1000, show: bool=True):
        data = self.generate(n)

        # Get artificial colors
        bp_rp = data['BP_mag'] - data['RP_mag']

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.scatter(
            bp_rp,
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