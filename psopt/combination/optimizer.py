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

    config = {
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
    }  # type typing.Dict[typing.Text, typing.Any]

    def __init__(self, obj_func, candidates, constraints=None, **kwargs):
        super().config.update(__class__.config)
        super().__init__(obj_func=obj_func,
                         candidates=candidates,
                         constraints=constraints,
                         **kwargs)

    def _init_particles(self):

        super()._init_particles()

        # particles's velocities
        self._velocities = np.zeros((self.swarm_population, self.n_candidates))

        # selection probabilities
        self._probabilities = (
            np.ones((self.swarm_population, self.n_candidates))
            / self.n_candidates
         )

    def _generate_particles(self, i: int, seed: int) -> typing.List[int]:

        np.random.seed(seed)
        candidates = np.random.choice(
            [j for j in range(self.n_candidates)],
            self.selection_size,
            p=self._probabilities[i],
            replace=False
        )

        return [1 if x in candidates else 0 for x in range(self.n_candidates)]

    def _update_particles(self, **kwargs):

        position = np.array(self._particles[-2]["position"])

        # velocities Update
        self._velocities = self._w * self._velocities

        particle_best_comp = self._particles_best[-2]["position"] - position

        self._velocities += (
            self._c1 *
            np.random.random(particle_best_comp.shape) *
            particle_best_comp
        )

        global_best_comp = (
            np.tile(
                self._global_best[-2]["position"],
                (self.swarm_population, 1)
            )
            - position
        )

        self._velocities += (
            self._c2 *
            np.random.random() *
            global_best_comp
        )

        # velocity clamping
        self._velocities[self._velocities > self.n_candidates / 2] = (
            self.n_candidates / 2
        )

        self._probabilities = 1 / (1 + np.exp((- self._velocities)))

        self._probabilities /= np.sum(self._probabilities, axis=1)[:, None]

        self._particles[-1]["position"] = (
            kwargs["pool"].starmap(
                self._generate_particles,
                zip(
                    list(range(self.swarm_population)),
                    kwargs["seed"]
                )
            )
        )

    def _get_particle(self, position):
        return [self._candidates[x]
                for x in range(self.n_candidates)
                if position[x] == 1]

    def _get_labels(self, position):
        return [self.labels[x]
                for x in range(self.n_candidates)
                if position[x] == 1]
