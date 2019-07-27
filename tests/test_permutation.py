import pytest
from psopt.permutation import Permutation
from permutation import coSum, TSP


seed = 0


class TestcoSum:

    cosum = coSum()
    solver = Permutation(
        coSum.obj_func,
        cosum.candidates,
        cosum.constraint,
        metrics="l2"
    )
    solution = solver.minimize(
        max_iter=5,
        selection_size=cosum.selection_size,
        population=20,
        seed=seed
    )

    @pytest.mark.parametrize('solution', [solution])
    def test_solution(self, solution):
        assert solution.value < 5.5

    @pytest.mark.parametrize('solution', [solution])
    def test_max_iter(self, solution):
        assert solution.meta["max iterations"] == 5
        assert solution.results["iterations"] <= 5

    @pytest.mark.parametrize('solution', [solution])
    def test_feasibilty(self, solution):
        assert solution.results["feasible"]

    @pytest.mark.parametrize('solution', [solution])
    def test_metrics(self, solution):
        assert "l2" in solution.history.keys()

    @pytest.mark.parametrize('solver', [solver])
    def test_constraint_acceptance(self, solver):
        assert solver.constraints


class TestTSP:

    tsp = TSP()
    solver = Permutation(tsp.obj_func, tsp.candidates)
    solution = solver.minimize(
        max_iter=50,
        population=15,
        seed=seed
    )

    @pytest.mark.parametrize('solution', [solution])
    def test_solution(self, solution):
        assert solution.value <= 6371
