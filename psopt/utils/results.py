import csv
import json
import os
import typing

import matplotlib.pyplot as plt

clsResults = typing.TypeVar("clsResults", bound="Results")


class Results():
    """Class to store the optimization results"""

    class _History(dict):
        def plot(self,
                 y: typing.Union[typing.Text, typing.List[typing.Text]],
                 x: typing.Optional[typing.Text] = None):

            if isinstance(y, str):
                y = [y]

            if not x:
                x_values = range(len(self[y[0]]))
                x = "iteration"
            else:
                x_values = self[x]
                x = x

            for key in y:
                plt.plot(x_values, self[key], label=key.replace("_", " "))

            plt.xlabel(x)
            plt.legend()
            plt.show()

    def __init__(self, *args, **kwargs):
        self.history = self._History()
        self.meta = dict()
        self.results = dict()

    @property
    def solution(self):
        return self.results["solution"]

    @property
    def value(self):
        return self.results["value"]

    def load_meta(self, directory: typing.Text):
        """Loads a .json metafile and stores it in the Result.meta attribute

        Args:
            directory: str containing the path to file
        """
        filename = os.path.join(directory, "meta.json")
        with open(filename, "r") as _file:
            temp_dict = json.load(_file)
            self.meta.update(temp_dict)

    def load_history(self, directory: typing.Text):
        """Loads a .csv containing the progress of an optimization process
        and stores it in the Result.history attribute

        Args:
            directory: str containing the path to file
        """
        temp_dict = dict()
        filename = os.path.join(directory, "results.csv")
        with open(filename, "r") as _file:
            rows = list(csv.DictReader(_file))
            header = list(rows[0].keys())
            for key in header:
                temp_dict[key] = [float(row[key]) for row in rows]
            self.history.update(temp_dict)

    @classmethod
    def from_folder(
        cls: typing.Type[clsResults],
        directory: typing.Text
    ) -> clsResults:
        """Instantiates Result object from files within the directory

        Args:
            directory: str containing the path to files

        Returns:
            a Results objects
        """
        instance = cls()
        instance.load_meta(directory)
        instance.load_history(directory)

        return instance
