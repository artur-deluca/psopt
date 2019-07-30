import pytest
from psopt.combination import Combination
from combination import coSum, HSAT


seed = 0


class TestcoSum:

    cosum = coSum(seed=seed)
    solver = Combination(
        coSum.obj_func,
        cosum.candidates,
        cosum.constraint,
    )
    solution = solver.minimize(
        selection_size=cosum.selection_size,
        population=20,
        threshold=cosum.threshold,
        seed=seed
    )

    @pytest.mark.parametrize("solution", [solution])
    @pytest.mark.parametrize("instance", [cosum])
    def test_solution(self, solution, instance):
        assert solution.value <= sum(sorted(instance.candidates)[:instance.selection_size])

    @pytest.mark.parametrize("solution", [solution])
    def test_feasibilty(self, solution):
        assert solution.results["feasible"]

    @pytest.mark.parametrize("solver", [solver])
    def test_constraint_acceptance(self, solver):
        assert solver.constraints


class TestHSAT:

    hsat = HSAT(seed=seed)
    solver = Combination(hsat.obj_func, hsat.candidates)
    solution = solver.minimize(
        selection_size=hsat.selection_size,
        threshold=hsat.threshold,
        seed=seed
    )

    @pytest.mark.parametrize("solution", [solution])
    def test_solution(self, solution):
        assert solution.value == 0

    @pytest.mark.parametrize("solution", [solution])
    @pytest.mark.parametrize("instance", [hsat])
    def test_solution_size(self, solution, instance):
        assert len(solution.solution) == instance.selection_size
