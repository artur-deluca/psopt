import random


def get_seeds(size):
    seeds = random.sample(list(range(size * 1000)), size)
    return seeds
