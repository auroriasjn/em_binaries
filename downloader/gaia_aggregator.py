import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from typing import List

class GaiaAggregator:
    def __init__(
        self,
        file_names: List[str] = ["clustercat.dat", "escapecat.dat"],
    ):
        """
        This class is just meant to aggregate some of the weird data that
        was achieved from the initial Pleiades fit code.
        """
        self.file_names = file_names

        # Aggregate data from files
        self.data = self._agg_files()
        self._clean_data()

    def _agg_files(self) -> pd.DataFrame:
        dfs = []
        for file_name in self.file_names:
            # Read manually and clean header
            with open(file_name) as f:
                header = f.readline().strip().lstrip("#").strip().split()
            temp_data = pd.read_csv(file_name, delim_whitespace=True, names=header, comment="#", skiprows=1)
            dfs.append(temp_data)

        common_cols = set.intersection(*(set(df.columns) for df in dfs))
        dfs = [df[list(common_cols)] for df in dfs]

        data = pd.concat(dfs, ignore_index=True)
        data.reset_index(drop=True, inplace=True)

        return data
    
    def _clean_data(self) -> pd.DataFrame:
        assert 'BpRp' in self.data.columns, "BpRp column not found in data."
        assert 'G' in self.data.columns, "G column not found in data."

        # Remove extraneous outlier values
        self.data = self.data[
            (self.data['BpRp'] != -99.99) & (self.data['G'] != -99.99)
        ]
        return self.data

    def get_data(self):
        return self.data
    