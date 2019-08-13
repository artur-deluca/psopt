import csv
import os
import typing

import matplotlib.pyplot as plt

clsResults = typing.TypeVar("clsResults", bound="Results")


class Results:
    """Class that stores the optimization results"""

    class History(dict):
        def plot(self, y, x=None):
            # type: (, typing.Union[typing.Text, typing.List[typing.Text]], typing.Optional[typing.Text])
            """Plot any given measured characteristic recorded througout the optimization procedure

            Args:
                y (list or str): name of the metrics at the y-axis
                x (str): name of the metrics for the x-axis.
                    If empty, the horizontal axis will be the ``iterations``"""

            plt.figure(figsize=(8, 6))
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
        self._history = self.History()
        self.meta = dict()
        self.results = dict()

    @property
    def history(self):
        """A History object containing the recorded values of the global-best,
            iteration best and any other metric specified"""
        return self._history

    @property
    def solution(self):
        """The solution found of within the given set of candidates"""
        return self.results["solution"]

    @property
    def value(self):
        """The value associated with the solution found"""
        return self.results["value"]

    def load_history(self, directory: typing.Text, delete: bool):
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
        if delete:
            os.remove(filename)

    @classmethod
    def load(cls: typing.Type[clsResults], directory: typing.Text) -> clsResults:
        """Instantiates Result object from files within the directory

        Args:
            directory: str containing the path to files

        Returns:
            a Results objects
        """
        instance = cls()
        instance.load_history(directory, delete=False)

        return instance
