"""
===========================
Constrained sum of elements
===========================

From a shuffled set of integers, find the numbers that minimize the sum of
5 (five) elements, so that the result is also an odd number

In this case, since the objective function is the sum of the elements, the order doesn't matter, so we use the ``Combination`` optimizer.
"""

import random
from psopt import Combination


def main():
    # define objective function: f([a, b, c, ...]) = a + b + c + ...
    def obj_func(x):
        return sum(x)

    # list of possible candidates
    random.seed(0)
    candidates = random.sample(list(range(1, 12)), 11)

    # constraint: sum of values cannot be even
    def mod(x):
        return sum(x) % 2

    constraint = {"fn": mod, "type": "==", "value": 1}

    # instantiate the optimizer
    opt = Combination(obj_func, candidates, constraints=constraint)

    # define a threshold of acceptance for early convergence
    threshold = 15

    # minimize the obj function
    results = opt.minimize(selection_size=5, threshold=threshold, seed=0, verbose=1)
    print("Solution: ", results.solution)

    results.history.plot(["global_best", "iteration_best"])


if __name__ == "__main__":
    main()
