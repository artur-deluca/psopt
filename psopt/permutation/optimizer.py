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

    config = {
        "w": 0.2,
        "c1": 0.8,
        "c2": 0.8,
    }  # type typing.Dict[typing.Text, typing.Any]

    def __init__(self, obj_func, candidates, constraints=None, **kwargs):
        super().config.update(__class__.config)
        super().__init__(obj_func=obj_func,
                         candidates=candidates,
                         constraints=constraints,
                         **kwargs)

    def _update_particles(self, **kwargs):
        self._particles[-1]["position"] = (
            kwargs["pool"].starmap(
                self._multi_position,
                zip(
                    list(range(self.swarm_population)),
                    kwargs["seed"]
                )
            )
        )

    def _generate_particles(self, i: int, seed: int) -> typing.List[int]:

        np.random.seed(seed)
        candidates = np.random.permutation(np.arange(self.n_candidates))
        candidates = candidates[:self.selection_size]

        return candidates

    def _get_particle(self, position: typing.List[int]):
        return [self._candidates[x] for x in position]

    def _get_labels(self, position: typing.List[int]):
        return [self.labels[i] for i in position]

    def _multi_position(self, i: int, seed: int):

        np.random.seed(seed)

        # retrieving positions for the calculation
        position = self._particles[-2]["position"][i]
        p_best = self._particles_best[-2]["position"][i]
        g_best = self._global_best[-2]["position"]

        if np.random.random() < self._w:
            position = self._mutate(position)
        if np.random.random() < self._c1:
            position = self._crossover(position, p_best)
        if np.random.random() < self._c2:
            position = self._crossover(position, g_best)

        position = position.astype(int)

        if (len(np.unique(position)) != self.selection_size):
            self._logger.warning(
                "Particle with repeated items, re-initializing it"
            )
            position = self._generate_particles(0, seed)
        return position

    def _mutate(self, p: typing.List[int]) -> typing.List[int]:
        """Performs a swap mutation with the remaining available itens"""
        if len(p) > 1:
            # get random slice
            _slice = np.random.permutation(self.selection_size)[:2]
            start, finish = min(_slice), max(_slice)
            p_1 = np.append(p[0:start], p[finish:])
            p_2 = list(set(range(self.n_candidates)) - set(p_1))
            p[start:finish] = np.random.choice(
                p_2,
                size=len(p[start:finish]),
                replace=False
            )
        return p

    @staticmethod
    def _crossover(p_1: typing.List[int],
                   p_2: typing.List[int]) -> typing.List[int]:
        """Performs the PTL Crossover between two sequences"""
        indexes = list(range(len(p_1)))
        if len(p_1) == len(p_2) and len(p_1) > 1:
            # get random slice from the first array
            _slice = np.random.permutation(indexes)[:2]
            start, finish = min(_slice), max(_slice)
            p_1 = p_1[start:finish]

            # remove from the second array the values found in
            # the slice of the first array

            p_2 = np.array([x for x in p_2 if x in (set(p_2) - set(p_1))])
            if len(p_2) + len(p_1) > len(indexes):
                p_2 = p_2[:len(indexes) - len(p_1)]

            # create the two possible combinations
            p_1, p_2 = np.append(p_1, p_2), np.append(p_2, p_1)
        return [p_1, p_2][np.random.randint(0, 2)]
