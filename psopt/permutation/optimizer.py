import numpy as np
import multiprocess
import time

from psopt.commons import Optimizer
from psopt.utils import evaluate_constraints


class PermutationOptimizer(Optimizer):
	"""Particle swarm permutation optimizer to find the best permutation of candidates
	Implementation based on:
		Pan, Q.-K., Fatih Tasgetiren, M., and Liang, Y.-C. (2008). A discrete particle swarm optimization algorithm for the no-wait flowshop scheduling problem.

	Parameters
	----------
		obj_func: function
			objective function to be optimized. Must only accept the candidates as input.
			If the inherited structure does not allow it, use `functools.partial` to create partial functions

		candidates: list
			list of inputs to pass to the objective function

		constraints: function or sequence of functions, optional
			functions to limit the feasible solution space

	Example
	-------

		>>> candidates = [2,4,5,6,3,1,7]

		>>> # e.g. obj_func([a, b, c, d, e]) ==> a + b/2 + c/3 + d/4 + e/5
		>>> def obj_func(x): return sum([a / (i+1) for i, a in enumerate(x)])

		>>> # constraint: sum of values cannot be greater than 16
		>>> constraint = {"fn":sum, "type":">", "value":16}

		>>> # minimize the obj function
		>>> opt = PermutationOptimizer(obj_func, candidates, constraints=constraint)
		>>> sol = opt.minimize(selection_size=5)

	"""

	config = {
		"w": 0.2,
		"c1": 0.8,
		"c2": 0.8,
	}

	def __init__(self, obj_func, candidates, constraints=None, labels=None, **kwargs):
		Optimizer.config.update(__class__.config)
		Optimizer.__init__(self, obj_func=obj_func, candidates=candidates, constraints=constraints, labels=labels, **kwargs)

	def _optimize(self):

		start = time.time()

		# set default exit flag
		exit_flag = 0

		# create pool for parallel processing
		pool = multiprocess.Pool()

		# initialize "storage" arrays
		self._init_particles()

		# initialize particles
		iteration = 0
		self._particles[-1]["position"] = [self._generate_particles() for _ in range(self.swarm_population)]

		# optimizing
		while(1 and iteration < self._max_iter):

			self._particles.append(self._template_position)
			self._particles_best.append(self._template_position)
			self._global_best.append(self._template_global)
			self._w = self._update_w(iteration)

			results = pool.map(self._multi_obj_func, list(range(self.swarm_population)))
			results = list(map(list, zip(*results)))

			self._particles[-2]["value"] = np.array(results[0])
			self._particles_best[-2]["value"] = np.array(results[1])
			self._particles_best[-2]["position"] = results[2]

			# for early stopping use
			last_best = self._global_best[-2]["position"]

			# update Global Best
			if (self._global_best[-2]["value"] < max(self._particles_best[-2]["value"])):

				early_stop_counter = 0  # clear counter since new global best was found

				self._global_best[-2]["value"] = max(self._particles_best[-2]["value"])
				self._global_best[-1]["value"] = self._global_best[-2]["value"]
				self._global_best[-2]["position"] = self._particles_best[-2]["position"][self._particles_best[-2]["value"].argmax()]

				if self._global_best[-2]["value"] >= self._threshold:
					exit_flag = 2
					break

			else:
				self._global_best[-1]["value"] = self._global_best[-2]["value"]
				self._global_best[-2]["position"] = self._global_best[-3]["position"]

				# it may have the same value and not the same position
				if (last_best == self._global_best[-2]["position"]).all():
					early_stop_counter += 1
					if early_stop_counter >= self._early_stop:
						exit_flag = 1
						break
				else:
					early_stop_counter = 0

			self._logger.info('\rIteration %d gbest = %f and iteration best = %f' % (iteration, self._m * self._global_best[-2]["value"], self._m * max(self._particles[-2]["value"])))

			# update particles position
			positions = pool.map(self._multi_position, list(range(self.swarm_population)))
			self._particles[-1]["position"] = positions

			for i in range(self.swarm_population):
				if (len(np.unique(self._particles[-1]["position"][i])) != self.selection_size):
					tempVar = np.random.permutation(self.selection_size)
					self._particles[-1]["position"][i] = tempVar[0:self.selection_size]

			if not self._record and iteration > 2:
				self._particles.pop(0)
				self._particles_best.pop(0)
				self._global_best.pop(0)

			# stop criteria

			unique = np.unique(self._particles[-1]['position'])

			if len(unique) == self.swarm_population - 1:
				exit_flag = 3
				break
			# if for loop is not broken
			else:
				iteration = iteration + 1
				continue
			# if for loop is broken, break the while loop as well
			break

		# store results
		solution = self._global_best[-2]["position"]
		solution_value = self._m * self._global_best[-2]["value"]
		elapsed_time = time.time() - start

		self._exit(exit_flag)

		if evaluate_constraints(self.constraints, self._get_particle(solution)) > 0:
			self._logger.info("The algorithm was unable to find a feasible solution")

		self._logger.info("Elapsed time {}".format(elapsed_time))
		self._logger.info("{} iterations".format(iteration))
		self._logger.info("Best selection: {}".format([self.labels[i] for i in solution]))
		self._logger.info("Best evaluation: {}".format(solution_value))

		return solution

	def _generate_particles(self):

		candidates = np.random.permutation(np.arange(self.n_candidates))
		candidates = candidates[:self.selection_size]

		return candidates

	def _get_particle(self, position):
		return [self._candidates[x] for x in position]

	def _multi_position(self, i):
		"""Calculates the new position for each particle in the swarm"""

		# retrieving positions for the calculation
		position = np.copy(self._particles[-2]["position"][i])
		p_best = np.copy(self._particles_best[-2]["position"][i])
		g_best = np.copy(self._global_best[-2]["position"])

		if np.random.random() < self._w:
			position = self._mutate(position)
		if np.random.random() < self._c1:
			position = self._crossover(position, p_best)
		if np.random.random() < self._c2:
			position = self._crossover(position, g_best)
		position = position.astype(int)

		return position

	def _mutate(self, p):
		"""Performs a swap mutation with the remaining available itens"""
		indexes = list(range(len(p)))
		if len(p) > 1:
			# get random slice
			_slice = np.random.permutation(indexes)[:2]
			start, finish = min(_slice), max(_slice)

			p_1 = np.append(p[0:start], p[finish:])
			p_2 = list(set(list(range(self.n_candidates))) - set(p_1))
			p[start:finish] = np.random.choice(p_2, size=len(p[start:finish]))
		return p

	def _crossover(self, p_1, p_2):
		"""Performs the PTL Crossover between two sequences"""
		indexes = list(range(len(p_1)))
		if len(p_1) == len(p_2) and len(p_1) > 1:
			# get random slice from the first array
			_slice = np.random.permutation(indexes)[:2]
			start, finish = min(_slice), max(_slice)
			p_1 = p_1[start:finish]

			# remove from the second array the values found in the slice of the first array

			p_2 = np.array([x for x in p_2 if x in (set(p_2) - set(p_1))])
			if len(p_2) + len(p_1) > len(indexes):
				p_2 = p_2[:len(indexes) - len(p_1)]

			# create the two possible combinations
			p_1, p_2 = np.append(p_1, p_2), np.append(p_2, p_1)
		return [p_1, p_2][np.random.randint(0, 2)]
