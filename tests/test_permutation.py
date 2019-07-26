import pytest
from psopt.permutation import Permutation


# a simple buggy test for reference
def test_solution():
    obj_func = lambda x: sum([a / (i + 1) for i, a in enumerate(x)])
    choices = list(range(1, 11))
    solution = Permutation(obj_func, choices).minimize(
        max_iter=5,
        selection_size=5,
        population=20,
        seed=0
    )

    assert solution.value < 5.5
