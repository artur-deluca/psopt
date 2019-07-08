import functools
import multiprocessing
import numpy as np
import time

from support import create_batches, order_batches, obj_func


class PSOptimizer:
	"""
	Description:
		Particle swarm discrete optimizer
	Args:
		w: float, Default 0.5
			weights
		c1: float, Default 0.7
			particle confidence factor
		c2: float, Default 2.5
			swarm confidence factor
		coverage_factor: float, Default 0.3
			represents how many particles will be generated in comparison in relation with the selection pool
		max_iter: int, Default 100
			maximum number of iterations allowed
		min_size: int, Default 1
			minimum number of variables selected per particle
	"""

	def __init__(self, config={}):

		default_config = {
			"w": 0.5,
			"c1": 1,
			"c2": 2.5,
			"coverage_factor": 0.4,
			"max_iter": 100,
			"min_size": 1
		}

		for field in default_config:
			if field in config.keys():
				setattr(self, field, config[field])
			else:
				setattr(self, field, default_config[field])

		# TODO: implement a proper FLAG system
		self._EXEC_LEVEL = "NOTSET"

	def set_EXEC_LEVEL(self, level):
		self._EXEC_LEVEL = level

	def optimize(self, obj_func, n_candidates, constraints=[], verbose=False, selection_size=2):
		'''
		Particle swarm optimizer utilizing a probability thershold
		Args:
			n_candidates:

		Returns:
			SelectionValue: float
				value of the best combination
			iteration: int
				iteration in which the program ended
			bestComb: list
				best combination of batches
		'''

		# TODO: implement system without fixed number of selections
		self.selection_size = selection_size

		start = time.time()
		self.swarm_population = int(n_candidates * self.coverage_factor)

		# create pool for parallel processing
		pool = multiprocessing.Pool()

		iteration = 0
		if type(self.w) == list:
			w_min, w_max = min(self.w), max(self.w)
			w = lambda i: (w_max - ((w_max - w_min) * (iteration / self.max_iter)))  # noqa: E731
		else:
			w_input = self.w
			w = lambda: w_input  # noqa: E731

		# order batches
		indexes = np.array(range(n_candidates))

		listOfIndexes = list(range(n_candidates))
		upperBoundary = listOfIndexes[-1]  # noqa: F841

		#  initialize "storage" arrays
		self._intialize_arrays(n_candidates)

		# initialize particles
		self._generate_particles(constraints=constraints, pool_size=n_candidates)

		# TODO: replace self._mPosition with one hot encoding operations when necessary

		# optimizing
		while(1 and iteration < self.max_iter - 1):

			args_ = list(
				zip(
					[obj_func] * self.swarm_population,
					list(range(self.swarm_population)),
					[iteration] * self.swarm_population
				)
			)
			results = pool.starmap(self._parallelize_obj_func, args_)
			results = list(map(list, zip(*results)))

			self._mParticles[iteration]["value"] = np.array(results[0])
			self._mParticlesbest[iteration]["value"] = np.array(results[1])
			self._mParticlesbest[iteration]["position"] = results[2]

			if self._EXEC_LEVEL == "DEBUG":
				self._diversity(self._mParticlesbest[iteration]["position"])

			# Update Global Best
			if (self._globalBest[iteration]["value"] < max(self._mParticlesbest[iteration]["value"])):

				self._globalBest[iteration]["value"] = max(self._mParticlesbest[iteration]["value"])
				self._globalBest[iteration + 1]["value"] = max(self._mParticlesbest[iteration]["value"])
				self._globalBest[iteration]["position"] = self._mParticlesbest[iteration]["position"][self._mParticlesbest[iteration]["value"].argmax()]

			else:
				self._globalBest[iteration + 1]["value"] = self._globalBest[iteration]["value"]
				self._globalBest[iteration]["position"] = self._globalBest[iteration - 1]["position"]
				self._globalBest[iteration + 1]["position"] = self._globalBest[iteration]["position"]

			bestComb = list(indexes[np.where(np.array(self._globalBest[iteration]["position"]) == 1)[0].tolist()])

			if verbose:
				print('\rIteration %d gbest = %f and iteration best = %f' % (iteration, self._globalBest[iteration]["value"], max(self._mParticles[iteration]["value"])))

			# mVelocities Update
			self._mVelocities = w() * self._mVelocities

			particle_best_comp = np.reshape(list(self._mParticlesbest[iteration]["position"]), (self.swarm_population, n_candidates)) - self._mPositions
			particle_best_comp = self._interpolate_matrix(particle_best_comp)

			self._mVelocities += self.c1 * np.random.random() * particle_best_comp

			global_best_comp = np.tile(self._globalBest[iteration]["position"], (self.swarm_population, 1)) - self._mPositions
			global_best_comp = self._interpolate_matrix(global_best_comp)

			self._mVelocities += self.c2 * np.random.random() * global_best_comp

			self._mProbabilities = 1 / (1 + np.exp((- self._mVelocities)))

			self._mProbabilities /= np.sum(self._mProbabilities, axis=1)[:, None]

			del particle_best_comp
			del global_best_comp

			if self._EXEC_LEVEL == "DEBUG":
				self._diversity(self._mProbabilities)

			# Postion Update
			self._generate_particles(constraints=constraints, pool_size=n_candidates, iteration=iteration + 1)

			# stop criteria
			count = 0

			# think of a more elegant way to the stop criteria
			for i in range(0, self.swarm_population - 1):
				if ((self._mPositions[i] == self._mPositions[i + 1]).all()):
					count = count + 1
			if ((count == self.swarm_population - 1) or (iteration >= self.max_iter - 1)):
				print('\n******Iteration completed******\n')
				break
			# if for loop is not broken
			else:
				iteration = iteration + 1
				continue
			# if for loop is broken, break the while loop as well
			break

		# store results
		Selection = self._globalBest[iteration]["position"]  # noqa: F841
		SelectionValue = self._globalBest[iteration]["value"]
		elapsed_time = time.time() - start  # noqa: F841

		if self._EXEC_LEVEL == 'MESH':
			positions, values = list(map(list, zip(*list(map(lambda x: (x['position'], x['value']), self._mParticles)))))
			np.savez('opt_position_log', positions=positions, values=values)

		if verbose:
			print("Elapsed time %f" % (elapsed_time))
			print("Iteration %d" % (iteration))
			print("Best selection: ", Selection)
			print("Best evaluation: ", SelectionValue)

		return SelectionValue, iteration, bestComb

	def _parallelize_obj_func(self, obj_func, i, iteration):

		evaluation = obj_func(self._mParticles[iteration]["position"][i])

		# self._mResults[list(self._mParticles[iteration]["position"][i]).count(1) - 1].append(self._mParticles[iteration]["value"][i]) # suspended

		evaluations = np.array(list(map(lambda k: k[i], list(map(lambda x: x["value"], self._mParticles[:iteration + 1])))))

		if evaluation > max(evaluations):
			best = evaluation
			best_selection = [int(k) for k in self._mPositions[i]]
		else:
			best = max(evaluations)
			# different from the first case
			# best_selection = self._mParticles[best.argmax()]["position"][0]
			best_selection = self._mPositions[i]

		return [evaluation, best, best_selection]

	def _intialize_arrays(self, pool_size):
		# initialize Arrays best
		self._mProbabilities = np.ones((self.swarm_population, pool_size)) / pool_size

		# self._mResults = [[] for _ in range(int(pool_size))] # suspended functionality

		#self._mSelProbabilities = (np.ones((1, pool_size)) / pool_size)[0]

		self._mPositions = np.zeros((self.swarm_population, pool_size))

		self._mVelocities = np.ones((self.swarm_population, pool_size))

		# mParticles[iteration][position, value][particle]
		self._mParticles = [
			{
				"position": [[] for _ in range(self.swarm_population)],
				"value": [0 for _ in range(self.swarm_population)]
			} for _ in range(self.max_iter + 1)
		]

		# mParticlesbest[iteration][position, value][particle]
		self._mParticlesbest = np.array([
			{
				"position": [np.array([]) for _ in range(self.swarm_population)],
				"value": [0 for _ in range(self.swarm_population)]
			} for _ in range(self.max_iter + 1)
		])

		self._globalBest = np.array([{"position": [], "value": 0} for _ in range(0, self.max_iter + 1)])  # globalBest[iteration][position,value]

	def _generate_particles(self, constraints, pool_size, iteration=0):
		for i in range(0, self.swarm_population):
			candidates = np.random.choice(
				[j for j in range(pool_size)],
				self.selection_size,
				p=self._mProbabilities[i],
				replace=True
			)

			if len(constraints) > 0:

				attend_constraints = self._evaluate_constraints(constraints, candidates)

				while (set(attend_constraints) != {True}):

					candidates = np.random.choice(
						[j for j in range(pool_size)],
						self.selection_size,
						p=self._mProbabilities[i],
						replace=True
					)

					attend_constraints = self._evaluate_constraints(constraints, candidates)

			self._mParticles[iteration]["position"][i] = candidates

			self._mPositions[i]
			for j in range(len(self._mPositions[i])):
				if j in candidates:
					self._mPositions[i][j] = 1
				else:
					self._mPositions[i][j] = 0

	@staticmethod
	def _diversity(array, only_sum=True):
		'''
		Helper function to estimate the diveristy of components among particles
		Args:
			array: list
			only_sum: boolean, default True
				print only the diversity metric
		'''
		array = np.array([np.array(i) for i in array])

		difference = list()
		for i in range(1, len(array)):
			difference.append(array[i] - array[i - 1])
		difference = np.mean(difference, axis=0)
		print("Diversity: {}".format(round(np.mean(difference), 3)))
		if not only_sum:
			print(np.around(difference, decimals=3))

	@staticmethod
	def _hamming(array, only_sum=True):
		'''
		Helper function to estimate the hamminton distance among particles
		Args:
			array: list
			only_sum: boolean, default True
				print only the diversity metric
		'''
		array = np.array([np.array(i) for i in array])

		difference = list()
		for i in range(1, len(array)):
			difference.append(np.count_nonzero(array[i]!=array[i - 1]))
		
		difference = np.mean(difference, axis=0)
		
		print("Hamming: {}".format(round(np.mean(difference), 3)))
		if not only_sum:
			print(np.around(difference, decimals=3))

	@staticmethod
	def _evaluate_constraints(constraints, candidates):
		if len(constraints) > 0:
			attend_constraints = [True]
		else:
			attend_constraints = list()
			for constraint in constraints:
				attend_constraints.append(constraint(candidates))
		return attend_constraints

	@staticmethod
	def _interpolate_matrix(matrix):
		matrix = np.array(
			list(
				map(
					lambda x: np.interp(
						list(range(len(x))),
						np.concatenate([[0], np.nonzero(x[1:-1])[0] + 1, [len(x) - 1]]),
						np.concatenate([[x[0]], x[1:-1][x[1:-1] != 0], [x[-1]]])
					), matrix)
			)
		)
		return matrix


if __name__ == '__main__':

	total_set = create_batches(30)
	selected = total_set[:2]
	batch_pool = total_set[2:]

	batch_pool_measurements = order_batches(batch_pool, orderby='mean')
	selected_measurements = order_batches(selected, orderby='mean')

	func = functools.partial(obj_func, selected_measurements, batch_pool_measurements)

	opt = PSOptimizer()

	opt.optimize(func, len(batch_pool_measurements), verbose=True)
