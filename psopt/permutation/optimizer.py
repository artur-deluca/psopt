import random
import typing

import numpy as np

from psopt.commons import Optimizer


class Permutation(Optimizer):
    """Solver to find an optimal permutation of candidates

    Implementation based on:
        Pan, Q.-K., Fatih Tasgetiren, M., and Liang, Y.-C. (2008)
            A discrete particle swarm optimization algorithm for
            the no-wait flowshop scheduling problem.

    Args:
        obj_func: objective function (or method) to be optimized.
            Must only accept the candidates as input.
            If the inherited structure does not allow it,
            use `functools.partial` to comply

        candidates: list of available candidates to the objective function

        constraints: function or list of functions to limit
            the feasible solution space

    Returns:
        Permutation optimizer object

    Example:

        >>> candidates = [2,4,5,6,3,1,7]
        >>> # e.g. obj_func([a, b, c, d, e]) ==> a + b/2 + c/3 + d/4 + e/5
        >>> def obj_func(x): return sum([a / (i+1) for i, a in enumerate(x)])
        >>> # constraint: sum of values cannot be greater than 16
        >>> constraint = {"fn":sum, "type":">", "value":16}
        >>> # minimize the obj function
        >>> opt = Permutation(obj_func, candidates, constraints=constraint)
        >>> sol = opt.minimize(selection_size=5)

    """

    _config = {
        "w": 0.2,
        "c1": 0.8,
        "c2": 0.8,
    }  # type typing.Dict[typing.Text, typing.Any]

    # ================== Initialization methods ======================

    def __init__(self, obj_func, candidates, constraints=None, **kwargs):
        super()._config.update(__class__._config)
        super().__init__(
            obj_func=obj_func, candidates=candidates, constraints=constraints, **kwargs
        )

    def _generate_particles(self, pool, seeds: typing.List[int]):
        params = [
            {
                "seed": x,
                "n_candidates": self.n_candidates,
                "swarm_population": self.swarm_population,
                "selection_size": self.selection_size,
            }
            for x in seeds
        ]

        self._particles[-1]["position"] = pool.map(self._generate_candidate, params)

    @staticmethod
    def _generate_candidate(params: typing.Dict[str, int]) -> typing.List[int]:
        rand = random.Random(params["seed"])
        candidates = rand.sample(list(range(params["n_candidates"])), params["selection_size"])
        return candidates

    # ====================== Update methods ==========================

    def _update_components(self, pool, seeds):
        params = [
            {
                "logger": self._logger,
                "w": self._w,
                "c1": self._c1,
                "c2": self._c2,
                "seed": seed,
                "particle": particle,
                "pbest": pbest,
                "gbest": self._global_best[-2]["position"],
                "n_candidates": self.n_candidates,
                "selection_size": self.selection_size,
            }
            for seed, particle, pbest in zip(
                seeds,
                self._particles[-2]["position"],
                self._particles_best[-2]["position"],
            )
        ]

        self._particles[-1]["position"] = pool.map(self._update_candidate, params)

    @staticmethod
    def _update_candidate(params: typing.Dict[str, typing.Any]):

        rand = random.Random(params["seed"])

        # retrieving positions for the calculation
        particle = params["particle"].copy()
        pbest = params["pbest"].copy()
        gbest = params["gbest"].copy()

        if rand.random() < params["w"]:
            particle = Permutation._mutate(
                rand, particle, params["selection_size"], params["n_candidates"]
            )

        if rand.random() < params["c1"]:
            particle = Permutation._crossover(rand, particle, pbest)
        if rand.random() < params["c2"]:
            particle = Permutation._crossover(rand, particle, gbest)

        particle = list(map(int, particle))

        return particle

    @staticmethod
    def _mutate(
        rand, p: typing.List[int], selection_size: int, n_candidates: int
    ) -> typing.List[int]:
        """Performs a swap mutation with the remaining available itens"""
        if len(p) > 1:
            # get random slice
            _slice = rand.sample(list(range(selection_size)), 2)
            start, finish = min(_slice), max(_slice)
            p_1 = np.append(p[0:start], p[finish:])
            p_2 = list(set(range(n_candidates)) - set(p_1))
            p[start:finish] = rand.sample(p_2, len(p[start:finish]))
        return p

    @staticmethod
    def _crossover(rand, p_1: typing.List[int], p_2: typing.List[int]) -> typing.List[int]:
        """Performs the PTL Crossover between two sequences"""

        indexes = list(range(len(p_1)))
        if len(p_1) == len(p_2) and len(p_1) > 1:
            # get random slice from the first array
            _slice = rand.sample(indexes, 2)
            start, finish = min(_slice), max(_slice)
            p_1 = p_1[start:finish]

            # remove from the second array the values found in
            # the slice of the first array

            p_2 = np.array([x for x in p_2 if x in (set(p_2) - set(p_1))])
            if len(p_2) + len(p_1) > len(indexes):
                p_2 = p_2[: len(indexes) - len(p_1)]

            # create the two possible combinations
            p_1, p_2 = np.append(p_1, p_2), np.append(p_2, p_1)
        return [p_1, p_2][rand.randint(0, 1)]

    # ===================== Retrival methods =========================

    def _get_particle(self, position: typing.List[int]):
        return [self._candidates[x] for x in position]

    def _get_labels(self, position: typing.List[int]):
        return [self.labels[i] for i in position]

        # ===================== Optimization methods =========================

    def maximize(self, selection_size=None, verbose=0, **kwargs):
        """Seeks the solution that yields the maximum objective function value

        Args:
            selection_size (int): number of candidates that compose a solution.

            verbose (int): controls the verbosity while optimizing

                0. Display nothing (default);
                1. Display statuses on console;
                2. Display statuses on console and store them in ``.logs``.

            w (float or sequence): The *inertia factor* controls the contribution of the previous movement.
                If a single value is provided, w is fixed, otherwise it
                linearly alters from min to max within the sequence provided.

            c1 (float): The *self-confidence factor* controls the contribution derived by the difference between
                a particle's current position and it's best position found so far.

            c2 (float): The *swarm-confidence factor* controls the contribution derived by the difference between
                a particle's current position and the swarm's best position found so far.

            population (float or int): Factor to cover the search space
                (e.g. 0.5 would generate a number of particles of half the search space).
                If `population` is greater than one, the population size will have its value assigned.

            max_iter (int): Maximum possible number of iterations (default 300).

            early_stop (int): Maximum number of consecutive iterations with no improvement 
                that the algorithm accepts without stopping (default ``max_iter``).

        Returns:
            a Result object containing the solution, metadata
                and stored metrics history
        """
        return super().maximize(
            selection_size=selection_size, verbose=verbose, **kwargs
        )

    def minimize(self, selection_size=None, verbose=0, **kwargs):
        """Seeks the solution that yields the minimum objective function value

        Args:
            selection_size (int): number of candidates that compose a solution.

            verbose (int): controls the verbosity while optimizing

                0. Display nothing (default);
                1. Display statuses on console;
                2. Display statuses on console and store them in ``.logs``.

            w (float or sequence): The *inertia factor* controls the contribution of the previous movement.
                If a single value is provided, w is fixed, otherwise it
                linearly alters from min to max within the sequence provided.

            c1 (float): The *self-confidence factor* controls the contribution derived by the difference between
                a particle's current position and it's best position found so far.

            c2 (float): The *swarm-confidence factor* controls the contribution derived by the difference between
                a particle's current position and the swarm's best position found so far.

            population (float or int): Factor to cover the search space
                (e.g. 0.5 would generate a number of particles of half the search space).
                If `population` is greater than one, the population size will have its value assigned.

            max_iter (int): Maximum possible number of iterations (default 300).

            early_stop (int): Maximum number of consecutive iterations with no improvement 
                that the algorithm accepts without stopping (default ``max_iter``).

        Returns:
            a Result object containing the solution, metadata
                and stored metrics history
        """
        return super().minimize(
            selection_size=selection_size, verbose=verbose, **kwargs
        )
