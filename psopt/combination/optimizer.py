import typing

import numpy as np

from psopt.commons import Optimizer


class Combination(Optimizer):
    """Solver to find the optimal combination of candidates

    Implementation based on:
        Unler, A. and Murat, A. (2010).
            A discrete particle swarm optimization method for
            feature selection in binary classification problems.

    Args:
        obj_func: objective function (or method) to be optimized.
            Must only accept the candidates as input.
            If the inherited structure does not allow it,
            use `functools.partial` to comply

        candidates: list of available candidates to the objective function

        constraints: function or list of functions to limit
            the feasible solution space

    Returns:
        Combination optimizer object

    Example:

        >>> candidates = [2,4,5,6,3,1,7]
        >>> # e.g. obj_func([a, b, c, d, e]) ==> a + b + c + d + e
        >>> def obj_func(x): return sum(x)
        >>> # constraint: sum of values cannot be even
        >>> def mod(x): return sum(x) % 2
        >>> constraint = {"fn":mod, "type":"==", "value":1}
        >>> # define a threshold of acceptance for early convergence
        >>> limit=15
        >>> # maximize the obj function
        >>> opt = Combination(obj_func, candidates, constraints=constraint)
        >>> sol = opt.maximize(selection_size=5, verbose=True, threshold=limit)

    """

    _config = {
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
    }  # type typing.Dict[typing.Text, typing.Any]

    # ================== Initialization methods ======================

    def __init__(self, obj_func, candidates, constraints=None, **kwargs):
        super()._config.update(__class__._config)
        super().__init__(
            obj_func=obj_func, candidates=candidates, constraints=constraints, **kwargs
        )

    def _init_storage_fields(self):

        super()._init_storage_fields()

        # particles's velocities
        self._velocities = np.zeros((self.swarm_population, self.n_candidates))

        # selection probabilities
        self._probabilities = (
            np.ones((self.swarm_population, self.n_candidates)) / self.n_candidates
        )

    def _generate_particles(self, pool, seeds: typing.List[int]):
        params = [
            {
                "seed": seed,
                "n_candidates": self.n_candidates,
                "swarm_population": self.swarm_population,
                "selection_size": self.selection_size,
                "probabilities": prob,
            }
            for seed, prob in zip(seeds, self._probabilities)
        ]

        self._particles[-1]["position"] = pool.map(self._generate_candidate, params)

    # ====================== Update methods ==========================

    @staticmethod
    def _generate_candidate(params) -> typing.List[int]:
        np.random.seed(params["seed"])
        candidates = np.random.choice(
            [j for j in range(params["n_candidates"])],
            params["selection_size"],
            p=params["probabilities"],
            replace=False,
        )

        return [1 if x in candidates else 0 for x in range(params["n_candidates"])]

    def _update_components(self, pool, seeds: typing.List[int]):

        position = np.array(self._particles[-2]["position"])

        # velocities Update
        self._velocities = self._w * self._velocities

        particle_best_comp = self._particles_best[-2]["position"] - position

        self._velocities += (
            self._c1 * np.random.random(particle_best_comp.shape) * particle_best_comp
        )

        global_best_comp = (
            np.tile(self._global_best[-2]["position"], (self.swarm_population, 1))
            - position
        )

        self._velocities += self._c2 * np.random.random() * global_best_comp

        # velocity clamping
        self._velocities[self._velocities > self.n_candidates / 2] = (
            self.n_candidates / 2
        )

        self._probabilities = 1 / (1 + np.exp((-self._velocities)))

        self._probabilities /= np.sum(self._probabilities, axis=1)[:, None]

        self._generate_particles(pool=pool, seeds=seeds)

    # ===================== Retrival methods =========================

    def _get_particle(self, position):
        return [
            self._candidates[x] for x in range(self.n_candidates) if position[x] == 1
        ]

    def _get_labels(self, position):
        return [self.labels[x] for x in range(self.n_candidates) if position[x] == 1]

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
