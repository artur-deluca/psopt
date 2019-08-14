import random


def get_seeds(size):
    seeds = random.sample(list(range(size * 1000)), size)
    return seeds


def reset_random_state(func):
    def wrapper_reset_random_state(*args, **kwargs):
        state = random.getstate()
        result = func(*args, **kwargs)
        random.setstate(state)
        return result
    return wrapper_reset_random_state
