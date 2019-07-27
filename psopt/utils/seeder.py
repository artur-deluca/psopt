import numpy as np


def get_seeds(size):
    seeds = list(np.random.randint(size * 100, size=size))
    return seeds
