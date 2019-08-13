
"""
===================================
Constrained ordered sum of elements
===================================

The objective function of this problem is:

.. math::

    f(x) = \\sum_{i=1}^{|x|}\\frac{x_i}{i}

From a shuffled set of integers, find the numbers that minimize the sum of fractions, so that the sum of the candidates doesn't not surpass 15.

In this case, the order of the numbers matter, so we use the ``Permutation`` optimizer
"""

import random
from psopt import Permutation


def main():
    # define objective function: f([a, b, c, ...]) = a/1 + b/2 + c/3 + ...
    def obj_func(x):
        return sum([a / (i + 1) for i, a in enumerate(x)])

    # seed the pseudo-random number generator
    seed = 5
    random.seed(seed)

    # list of shuffled candidates ranging from 1 to 50
    candidates = random.sample(list(range(1, 51)), 50)

    selection_size = 5

    # constraint: sum of values cannot be greater than 18
    constraint = {
        "fn": sum,
        "type": ">",
        "value": 18,
    }

    # instantiate the optimizer
    def min_sum(particles):
        return min([sum(i) for i in particles])

    opt = Permutation(obj_func, candidates, constraints=constraint, metrics=[min_sum, 'l2'])

    # define a threshold of acceptance for early convergence
    threshold = obj_func(sorted(candidates)[:selection_size])

    # minimize the obj function
    result = opt.minimize(
        selection_size=selection_size, verbose=1, threshold=threshold, population=20, seed=seed
    )

    result.history.plot("min_sum")
    result.history.plot("global_best")


if __name__ == "__main__":
    main()
