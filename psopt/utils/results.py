import csv
import matplotlib.pyplot as plt
import json
import os


class Results():

	class History(dict):
		def plot(self, key):
			plt.plot(self[key])
			plt.ylabel(key)
			plt.xlabel("Iteration")
			plt.show()

	def __init__(self, *args, **kwargs):
		self.history = self.History()
		self.meta = dict()
		self.solution = None

	def load_meta(self, directory):
		filename = os.path.join(directory, "meta.json")
		with open(filename, "r") as _file:
			temp_dict = json.load(_file)
			self.meta.update(temp_dict)

	def load_history(self, directory):
		temp_dict = dict()
		filename = os.path.join(directory, "results.csv")
		with open(filename, "r") as _file:
			rows = list(csv.DictReader(_file))
			header = list(rows[0].keys())
			for key in header:
				temp_dict[key] = [float(row[key]) for row in rows]
			self.history.update(temp_dict)

	@classmethod
	def from_folder(cls, directory):

		instance = cls()
		instance.load_meta(directory)
		instance.load_history(directory)

		return instance
