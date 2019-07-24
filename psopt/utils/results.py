import csv
import json
import os
import typing

import matplotlib.pyplot as plt


class Results():
    """Class to store the optimization results"""

    class _History(dict):
        def plot(self,
                 y: typing.Union[str, typing.List[str]],
                 x: typing.Optional[str] = None):

            if isinstance(y, str):
                y = [y]

            if not x:
                x = range(len(self[y[0]]))
                xlabel = "iteration"
            else:
                x = self[x]
                xlabel = x

            for key in y:
                plt.plot(x, self[key], label=key.replace("_", " "))

            plt.xlabel(xlabel)
            plt.legend()
            plt.show()

    def __init__(self, *args, **kwargs):
        self.history = self._History()
        self.meta = dict()
        self.solution = None

    def load_meta(self, directory: str):
        """Loads a .json metafile and stores it in the Result.meta attribute
        
        Args:
            directory: str containing the path to file
        """
        filename = os.path.join(directory, "meta.json")
        with open(filename, "r") as _file:
            temp_dict = json.load(_file)
            self.meta.update(temp_dict)

    def load_history(self, directory: str):
        """Loads a .csv containing the progress of an optimization process and stores it in the Result.history attribute

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
    def from_folder(cls, directory: str):
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
