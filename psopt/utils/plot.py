import matplotlib.pyplot as plt
import csv


def plot(data, key):
    with open(data, "r",) as file:
        rows = csv.DictReader(file)
        plt.plot([float(row[key]) for row in rows])
        plt.ylabel(key)
        plt.xlabel("Iteration")
        plt.show()
