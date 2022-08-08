# -*- coding: utf-8 -*-
"""Implementation of Dataset class to store measurements."""
import pandas as pd
import pint_pandas

from qcvv.config import raise_error


class Dataset:
    """Class to store the data measured during the calibration routines.
    It is a wrapper to a pandas DataFrame with units of measure from the Pint
    library.

    Args:
        quantities (dict): dictionary containing additional quantities that the user
                        may save other than the pulse sequence output. The keys are the name of the
                        quantities and the corresponding values are the units of measure.
    """

    def __init__(self, quantities=None):

        self.df = pd.DataFrame(
            {
                "MSR": pd.Series(dtype="pint[V]"),
                "i": pd.Series(dtype="pint[V]"),
                "q": pd.Series(dtype="pint[V]"),
                "phase": pd.Series(dtype="pint[deg]"),
            }
        )

        if quantities is not None:
            for name, unit in quantities.items():
                self.df.insert(0, name, pd.Series(dtype=f"pint[{unit}]"))

    def add(self, data):
        """Add a row to dataset.

        Args:
            data (dict): dictionary containing the data to be added.
                        Every key should have the following form:
                        ``<name>[<unit>]``.
        """
        import re

        from pint import UnitRegistry

        ureg = UnitRegistry()
        l = len(self)
        for key, value in data.items():
            name = key.split("[")[0]
            unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
            # TODO: find a better way to do this
            self.df.loc[l + l // len(list(data.keys())), name] = value * ureg(unit)

    def __len__(self):
        """Computes the length of the dataset."""
        return len(self.df)

    @classmethod
    def load_data(cls, folder, routine, format):
        """Load data from specific format.

        Args:
            folder (path): path to the output folder from which the data will be loaded
            routine (str): calibration routine data to be loaded
            format (str): data format. Possible choices are 'csv' and 'pickle'.

        Returns:
            dataset (``Dataset``): dataset object with the loaded data.
        """
        obj = cls()
        if format == "csv":
            file = f"{folder}/data/{routine}/data.csv"
            obj.df = pd.read_csv(file, header=[0, 1])
            obj.df = obj.df.pint.quantify(level=-1)
            obj.df.pop("Unnamed: 0_level_0")
        elif format == "pickle":
            file = f"{folder}/data/{routine}/data.pkl"
            obj.df = pd.read_pickle(file)
        else:
            raise_error(ValueError, f"Cannot load data using {format} format.")

        return obj

    def to_csv(self, path):
        """Save data in csv file.

        Args:
            path (str): Path containing output folder."""
        self.df.pint.dequantify().to_csv(f"{path}/data.csv")

    def to_pickle(self, path):
        """Save data in pickel file.

        Args:
            path (str): Path containing output folder."""
        self.df.to_pickle(f"{path}/data.pkl")