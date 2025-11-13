import matplotlib.pyplot as plt

from scipy.stats import uniform, norm

from typing import Tuple
from isochrones import get_ichrone
from isochrones.priors import GaussianPrior, SalpeterPrior, DistancePrior, FlatPrior
from isochrones.populations import StarFormationHistory, StarPopulation

class IsochroneSynthesizer:
    def __init__(
        self,
        mass_bounds: Tuple[float] = (0.8, 6.0),        # 1–5 Msun: good for red clump + AGB
        age_bounds: Tuple[float, float] = (1.0, 4.0), # Gyr; older than Pleiades but nice AGB
        feh_range: Tuple[float, float] = (-0.1, 0.1), # near-solar
        AV_range: Tuple[float, float] = (0.0, 0.1),
        max_distance: int = 3000,                     # if you want a generic field/open cluster
        fB: float = 0.2,
        gamma: float = 0.1
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