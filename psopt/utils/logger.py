import csv
import logging
import os
import typing

from datetime import datetime


class CustomLogger(logging.Logger):
    """Logger class that outputs .csv metrics logs and .json metafiles"""

    def __init__(self, name: typing.Text):
        super().__init__(name)
        self.timestamp = str(datetime.now())

    def set_files(self, file_path: typing.Text):

        self.file_path = os.path.join(file_path, self.timestamp)
        os.makedirs(self.file_path, exist_ok=True)

        self.file_results = os.path.join(self.file_path, "results.csv")

    def write_metrics(self, values: dict):
        self._write_csv(self.file_results, values)

    @staticmethod
    def _write_csv(path: typing.Text,
                   values: dict):

        file_exists = os.path.isfile(path)

        with open(path, "a+") as csv_file:

            writer = csv.DictWriter(csv_file, values.keys())

            if not file_exists:
                writer.writeheader()
            writer.writerow(values)

    def __getstate__(self):
        return None  # avoid unecessary pickling issues


def make_logger(name: typing.Text,
                verbose: typing.Union[int, bool]) -> CustomLogger:

    logger = CustomLogger(name)
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter("%(message)s"))

    if verbose == 1:
        # only set the console handler level
        consoleHandler.setLevel(logging.INFO)

    elif verbose == 2:
        # set the console handler level
        consoleHandler.setLevel(logging.INFO)

        # make sure that fileHandler logging directory exists
        os.makedirs(os.path.join(os.getcwd(), ".logs"), exist_ok=True)

        # add fileHandler in logger
        fileHandler = logging.FileHandler(".logs/logging.log")
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(fileHandler)

    else:
        consoleHandler.setLevel(logging.WARNING)

    logger.set_files(os.path.join(os.getcwd(), ".logs"))

    logger.addHandler(consoleHandler)

    return logger
