import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class GaiaAnalyzer:
    def __init__(
        self,
        file_names: str | list[str] = ["clustercat.dat", "escapecat.dat"],
    ):
        self.file_names = file_names

        # Open files
        if isinstance(self.file_names, str):
            self.file_names = [self.file_names]

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
    
    def plot_hr_diagram(self, show: bool=True):
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.scatter(
            self.data['BpRp'],
            self.data['G'],
            s=1,
            c='blue',
            alpha=0.5
        )
        ax.set_xlabel('BP - RP Color Index')
        ax.set_ylabel('G Mean Magnitude')
        ax.invert_yaxis()
        ax.set_title('Hertzsprung-Russell Diagram')

        if show:
            plt.show()
        
        return fig, ax
        