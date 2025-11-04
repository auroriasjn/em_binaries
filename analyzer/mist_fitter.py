# https://isochrones.readthedocs.io/en/latest/quickstart.html
# https://github.com/timothydmorton/isochrones/tree/master

import pandas as pd
import emcee

# We are leveraging the isochrone package here
from isochrones.catalog import StarCatalog
from isochrones.cluster import StarClusterModel
from isochrones import get_ichrone

class MISTFitter:
    def __init__(
        self,
        data: pd.DataFrame,
        age_range: tuple[float, float] = (6.0, 10.2),
        feh_range: tuple[float, float] = (-2.0, 0.5),
    ):
        # Load data
        self.data = data

        self.age_range = age_range
        self.feh_range = feh_range

        # Initialize MIST isochrone model
        self.mist = get_ichrone('mist')

    def _reformat_data(self, data: pd.DataFrame):
        catalog = StarCatalog(data, no_uncs=True)
        return catalog

    def fit_isochrone(self, data):
        # We know the Pleiades has an age of 75-140 Myr so we can just cheat here
        model = StarClusterModel(data)

    # Could potentially invoke https://github.com/mncavieres/mistfit
