import numpy as np
import warnings

from psopt.utils import evaluate_constraints, make_logger


class Optimizer:
	"""Optimizer template

	Parameters
	----------
		obj_func: function
			objective function to be optimized. Must only accept the candidates as input.
			If the inherited structure does not allow it, use `functools.partial` to create partial functions

		candidates: list
			list of inputs to pass to the objective function

		constraints: function or sequence of functions, optional
			functions to limit the feasible solution space

	"""

	config = {
		"w": 1,
		"c1": 1,
		"c2": 1,
		"population": 20,
		"max_iter": 300,
		"early_stop": None,
		"threshold": np.inf,
		"penalty": 1000,
	}

	def __init__(self, obj_func, candidates, constraints=None, labels=None, **kwargs):

		warnings.filterwarnings("ignore")

		self._candidates = candidates
		self._obj_func = obj_func
		self.n_candidates = len(candidates)
		self.labels = labels or candidates

		constraints = constraints or []

		if type(constraints) is dict:
			self.constraints = [constraints]
		else:
			self.constraints = constraints

	def maximize(self, selection_size=None, verbose=False, random_state=None, **kwargs):
		"""Seeks the candidates that yields the maximum objective function value

		Parameters
		----------
			selection_size: int, optional
				The number of candidates to compose a solution. If not declared, the total number of candidates will be used as the selection size

			verbose: bool, default False

			random_state: int, default None
				The seed of a pseudo random number generator

		Keywords
		--------
			w (inertia): float or sequence (float, float)
				It controls the contribution of the previous movement. If a single value is provided, w is fixed, otherwise it linearly alters from min to max within the sequence provided.

			c1 (self-confidence): float
				It controls the contribution derived by the difference between a particle's current position and it's best position found so far.

			c2 (swarm-confidence): float
				It controls the contribution derived by the difference between a particle's current position and the swarm's best position found so far.

			population: float or int
				Factor to cover the search space (e.g. 0.5 would generate a number of particles of half the search space). If `population` is greater than one, the population size will have its value assigned.

			max_iter: int, default 300
				Maximum possible number of iterations.

			early_stop: int, default None
				Maximum number of consecutive iterations with no improvement that the algorithm accepts without stopping. If None, it does not interfere the optimization.

		Returns
		-------
			solution : list
				The optimization solution represented as an ordered list of selected candidates.

			solution_value: float
				The resulting optimization value.
		"""
		# setup algorithm parameters
		self._set_params(selection_size=selection_size, f_min=False, verbose=verbose, random_state=random_state, **kwargs)

		return self._optimize()

	def minimize(self, selection_size=None, verbose=False, random_state=None, **kwargs):
		"""Seeks the candidate that yields the minimum objective function value

		Parameters
		----------
			selection_size: int, optional
				The number of candidates to compose a solution. If not declared, the total number of candidates will be used as the selection size

			verbose: bool, default False

			random_state: int, default None
				The seed of a pseudo random number generator

		Keywords
		--------
			w (inertia): float or sequence (float, float)
				It controls the contribution of the previous movement. If a single value is provided, w is fixed, otherwise it linearly alters from min to max within the sequence provided.

			c1 (self-confidence): float
				It controls the contribution derived by the difference between a particle's current position and it's best position found so far.

			c2 (swarm-confidence): float
				It controls the contribution derived by the difference between a particle's current position and the swarm's best position found so far.

			population: float or int
				Factor to cover the search space (e.g. 0.5 would generate a number of particles of half the search space). If `population` is greater than one, the population size will have its value assigned.

			max_iter: int, default 300
				Maximum possible number of iterations.

			early_stop: int, default None
				Maximum number of consecutive iterations with no improvement that the algorithm accepts without stopping. If None, it does not interfere the optimization.


		Returns
		-------
			solution : list
				The optimization solution represented as an ordered list of selected candidates.

			solution_value: float
				The resulting optimization value.
		"""
		# setup algorithm parameters
		self._set_params(selection_size=selection_size, f_min=True, verbose=verbose, random_state=random_state, **kwargs)

		return self._optimize()

	def _optimize(self):
		return

	def _multi_obj_func(self, i):

		particle = self._get_particle(self._particles[-2]["position"][i])

		evaluation = self._m * self._obj_func(particle)

		evaluation += self._penalty * (self._m * evaluate_constraints(self.constraints, particle))

		evaluations = np.array([x["value"][i] for x in self._particles[:-1]])

		if evaluation > max(evaluations):
			best = evaluation
			best_selection = self._particles[-2]["position"][i]
		else:
			best = max(evaluations)
			best_selection = self._particles[evaluations.argmax()]["position"][i]

		return [evaluation, best, best_selection]

	def _init_particles(self):

		self._template_position = {
			"position": [[] for _ in range(self.swarm_population)],
			"value": [-np.inf for _ in range(self.swarm_population)]
		}

		self._template_global = {"position": [], "value": -np.inf}

		self._velocities = np.zeros((self.swarm_population, self.selection_size))

		# particles[iteration][position, value][particle]
		self._particles = [self._template_position]

		# particles_best[iteration][position, value][particle]
		self._particles_best = [self._template_position]

		# global_best[iteration][position,value]
		self._global_best = [self._template_global]

	def _get_particle(self, particle):
		return

	def _set_params(self, selection_size, f_min, verbose, random_state, **kwargs):

		# set optimizer logger
		self._logger = make_logger(__name__, verbose)

		if random_state:
			np.random.seed(random_state)

		# record all iterations
		self._record = False if "record" not in kwargs else kwargs["record"]

		# transform the maximization problem into a minimization
		self._m = 1 - 2 * f_min

		# set the selection size
		self.selection_size = selection_size if selection_size else self.n_candidates

		# set the swarm and optimization parameters
		for field in __class__.config:
			if field in kwargs.keys():
				setattr(self, "_{}".format(field), kwargs[field])
			else:
				setattr(self, "_{}".format(field), __class__.config[field])

		# configure acceptable threshold
		if self._threshold != np.inf:
			self._threshold *= self._m

		# configure early stopping if is None
		self._early_stop = self._early_stop or self._max_iter

		# set the number of particles (population)
		if self._population > 1:
			self.swarm_population = int(self._population)
		else:
			self.swarm_population = int(self.n_candidates * self._population)

		# set the inertia function
		try:
			w_start, w_finish = tuple(self._w)
			self._update_w = lambda i: (w_finish - ((w_finish - w_start) * (i / self._max_iter)))  # noqa: E731

		except TypeError:
			self._update_w = lambda i: self._w  # noqa: E731

	def _exit(self, flag):

		exit_flag = {
			0: "Algortihm reached the maximum limit of {} iterations".format(self._max_iter),
			1: "Algorithm has not improved for {} consecutive iterations".format(self._early_stop),
			2: "Algorithm has reached the value threshold of {}".format(self._m * self._threshold),
			3: "Particles converged to a single solution"
		}

		print()
		self._logger.info("Iteration completed\n==========================")
		self._logger.info("Exit code {}: {}".format(flag, exit_flag[flag]))
