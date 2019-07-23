import csv
import logging
import json
import os
from datetime import datetime


def make_logger(name, verbose):

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
        fileHandler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fileHandler)

    else:
        consoleHandler.setLevel(logging.WARNING)

    logger.set_files(os.path.join(os.getcwd(), ".logs"))

    logger.addHandler(consoleHandler)

    return logger


class CustomLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.timestamp = str(datetime.now())

    def set_files(self, file_path):

        self.file_path = os.path.join(file_path, self.timestamp)
        os.makedirs(self.file_path, exist_ok=True)

        self.file_results = os.path.join(self.file_path, "results.csv")
        self.file_meta = os.path.join(self.file_path, "meta.json")
        self.file_position = os.path.join(self.file_path, "positions.csv")

    def write_metrics(self, dict_values):
        self._write_csv(self.file_results, dict_values)

    def write_positions(self, dict_values):
        self._write_csv(self.file_position, dict_values)

    def write_meta(self, dict_values):
        with open(self.file_meta, "a+") as json_file:
                json.dump(dict_values, json_file, indent=4)

    def _write_csv(self, path, dict_values):
        file_exists = os.path.isfile(path)

        with open(path, "a+") as csv_file:

            writer = csv.DictWriter(csv_file, dict_values.keys())

            if not file_exists:
                writer.writeheader()
            writer.writerow(dict_values)
