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

        self.file_name = os.path.join(self.file_path, "results.csv")
        self.file_meta = os.path.join(self.file_path, "meta.json")
        self.file_meta = os.path.join(self.file_path, "positions.csv")

    def write_metrics(self, dict_values):
        try:
            file_exists = os.path.isfile(self.file_name)

            with open(self.file_name, "a+") as csv_file:

                writer = csv.DictWriter(csv_file, dict_values.keys())

                if not file_exists:
                    writer.writeheader()
                writer.writerow(dict_values)

        except AttributeError as err:
            if str(err) == "'Metric' object has no attribute 'file_name'":
                pass
            else:
                raise err

    def write_meta(self, dict_values):
        try:
            with open(self.file_meta, "a+") as json_file:
                    json.dump(dict_values, json_file, indent=4)
        except AttributeError as err:
            if str(err) == "'Metric' object has no attribute 'file_meta'":
                pass
            else:
                raise err

    def write_positions(self, dict_values):
        try:
            file_exists = os.path.isfile(self.file_name)

            with open(self.file_name, "a+") as csv_file:

                writer = csv.DictWriter(csv_file, dict_values.keys())

                if not file_exists:
                    writer.writeheader()
                writer.writerow(dict_values)

        except AttributeError as err:
            if str(err) == "'Metric' object has no attribute 'file_position'":
                pass
            else:
                raise err
