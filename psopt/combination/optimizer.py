import numpy as np
import multiprocess
import time

from psopt.commons import Optimizer
from psopt.utils import evaluate_constraints


class CombinationOptimizer(Optimizer):
	"""Particle swarm combination optimizer to find the best combination of candidates
	Implementation based on:
		Unler, A. and Murat, A. (2010). A discrete particle swarm optimization method for feature selection in binary classification problems.

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

		>>> # e.g. obj_func([a, b, c, d, e]) ==> a + b + c + d + e
		>>> def obj_func(x): return sum(x)

		>>> # constraint: sum of values cannot be even
		>>> def mod(x): return sum(x) % 2
		>>> constraint = {"fn":mod, "type":"==", "value":1}

		>>> threshold=15 # define a threshold of acceptance for early convergence

		>>> # maximize the obj function
		>>> opt = CombinationOptimizer(obj_func, candidates, constraints=constraint)
		>>> sol = opt.maximize(selection_size=5, verbose=True, threshold=threshold)

	"""

	config = {
		"w": 0.7,
		"c1": 1.4,
		"c2": 1.4,
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
		self._particles[-1]["position"] = pool.map(self._generate_particles, list(range(self.swarm_population)))

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
				if (last_best == self._global_best[-2]["position"]):
					early_stop_counter += 1
					if early_stop_counter >= self._early_stop:
						exit_flag = 1
						break
				else:
					early_stop_counter = 0

			self._logger.info('\rIteration %d gbest = %f and iteration best = %f' % (iteration, self._m * self._global_best[-2]["value"], self._m * max(self._particles[-2]["value"])))

			Position = np.array(self._particles[-2]["position"])

			# velocities Update
			self._velocities = self._w * self._velocities

			particle_best_comp = self._particles_best[-2]["position"] - Position

			self._velocities += self._c1 * np.random.random(particle_best_comp.shape) * particle_best_comp

			global_best_comp = np.tile(self._global_best[-2]["position"], (self.swarm_population, 1)) - Position

			self._velocities += self._c2 * np.random.random() * global_best_comp

			# velocity clamping
			self._velocities[self._velocities > self.n_candidates / 2] = self.n_candidates / 2

			self._probabilities = 1 / (1 + np.exp((- self._velocities)))

			self._probabilities /= np.sum(self._probabilities, axis=1)[:, None]

			self._particles[-1]["position"] = pool.map(self._generate_particles, list(range(self.swarm_population)))

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
		elapsed_time = time.time() - start  # noqa: F841

		self._exit(exit_flag)

		if evaluate_constraints(self.constraints, self._get_particle(solution)) > 0:
			self._logger.info("The algorithm was unable to find a feasible solution")

		self._logger.info("Elapsed time {}".format(elapsed_time))
		self._logger.info("{} iterations".format(iteration + 1))
		self._logger.info("Best selection: {}".format([self.labels[i] for i in range(self.n_candidates) if solution[i] == 1]))
		self._logger.info("Best evaluation: {}".format(solution_value))

		return self._get_particle(solution)

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

		# particles[iteration][position, value][particle]
		self._particles = [self._template_position]

		# particles_best[iteration][position, value][particle]
		self._particles_best = [self._template_position]

		# global_best[iteration][position,value]
		self._global_best = [self._template_global]

		# particles's velocities
		self._velocities = np.zeros((self.swarm_population, self.n_candidates))

		# selection probabilities
		self._probabilities = np.ones((self.swarm_population, self.n_candidates)) / self.n_candidates

	def _generate_particles(self, i):

		candidates = np.random.choice(
			[j for j in range(self.n_candidates)],
			self.selection_size,
			p=self._probabilities[i],
			replace=False
		)

		return [1 if x in candidates else 0 for x in range(self.n_candidates)]

	def _get_particle(self, position):
		return [self._candidates[x] for x in range(self.n_candidates) if position[x] == 1]
