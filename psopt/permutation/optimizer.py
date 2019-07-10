import logging.config
import numpy as np
import os
import pathos.multiprocessing as multiprocessing
import time
import warnings

from psopt.utils import evaluate_constraints



class PermutationOptimizer:
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
			"population": 20,
			"max_iter": 300,
			"early_stop": None,
			"threshold": np.inf,
			"epsilon": 0,
			"penalty": 1000
	}


	def __init__(self, obj_func, candidates, constraints=[], labels=None, **kwargs):

		self._candidates = candidates
		self._obj_func = obj_func
		self.n_candidates = len(candidates)
		warnings.filterwarnings("ignore")
		
		if labels is None:
			self.labels = candidates
		else:
			self.labels = labels

		if type(constraints) is dict:
			self.constraints = [constraints]
		else:
			self.constraints = constraints
		
		if "record" in kwargs.keys():
			self.__record = kwargs["record"]
		else:
			self.__record = False

		log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log.conf')
		logging.config.fileConfig(log_file_path, disable_existing_loggers=False)

		self._logger = logging.getLogger()
		self._logger.name = __name__

	def maximize(self, selection_size=None, verbose=False, random_state=None, **kwargs):
		"""Seeks the best permutation of candidates that yields the maximum objective function value

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
		"""Seeks the best permutation of candidates that yields the minimum objective function value
		
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
		
		start = time.time()

		# set default exit flag
		exit_flag = 0

		# create pool for parallel processing
		pool = multiprocessing.Pool()
		
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
				
				early_stop_counter = 0 # clear counter since new global best was found

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
					
			self._logger.debug('\rIteration %d gbest = %f and iteration best = %f' % (iteration, self._m * self._global_best[-2]["value"], self._m * max(self._particles[-2]["value"])))
			
			# update particles position
			positions = pool.map(self._multi_position, list(range(self.swarm_population)))
			self._particles[-1]["position"] = positions

			for i in range(self.swarm_population):
				if (len(np.unique(self._particles[-1]["position"][i])) != self.selection_size):
					tempVar = np.random.permutation(self.selection_size)
					self._particles[-1]["position"][i] = tempVar[0:self.selection_size]

			if not self.__record and iteration > 2:
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

	def _multi_obj_func(self, i):
		
		particle = self._get_particle(self._particles[-2]["position"][i])

		evaluation = self._m  * self._obj_func(particle)

		evaluation += self._penalty * (self._m  * evaluate_constraints(self.constraints, particle))

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

	def _generate_particles(self):
		
		candidates = np.random.permutation(np.arange(self.n_candidates))
		candidates = candidates[:self.selection_size]

		return candidates

	def _set_params(self, selection_size, f_min, verbose, random_state, **kwargs):
		
		# quick fix
		# just a reminder that this sets the logging level for all instances of root as well
		if verbose:
			self._logger.handlers[0].setLevel(logging.DEBUG)
		else:
			self._logger.handlers[0].setLevel(logging.WARNING)

		if random_state:
			np.random.seed(random_state)

		# transform the maximization problem into a minimization
		self._m = 1 - 2 * f_min

		# set the selection size
		if not selection_size:
			self.selection_size = self.n_candidates
		else:
			self.selection_size = selection_size

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
		if not self._early_stop:
			self._early_stop = self._max_iter

		# set the number of particles (population)
		if self._population > 1:
			self.swarm_population = int(self._population)
		else:
			self.swarm_population = int(self.n_candidates * self._population)

		# set the inertia function
		try:
			inertia = tuple(self._w)
			w_min, w_max = min(inertia), max(inertia)
			self._update_w = lambda i: (w_max - ((w_max - w_min) * (i / self._max_iter)))  # noqa: E731
		
		except TypeError:
			self._update_w = lambda i: self._w  # noqa: E731

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

	def _mutate(self, p):
		"""Performs a swap mutation with the remaining available itens"""
		indexes = list(range(len(p)))
		if len(p) > 1:
			# get random slice
			_slice = np.random.permutation(indexes)[:2]
			start, finish = min(_slice), max(_slice)
			#p_1 = p[start:finish]
			p_1 = np.append(p[0:start], p[finish:])
			p_2 = list(set(list(range(self.n_candidates)))-set(p_1))
			p[start:finish] = np.random.choice(p_2, size=len(p[start:finish]))
		return p
	
	@staticmethod
	def _crossover(p_1, p_2):
		"""Performs the PTL Crossover between two sequences"""
		indexes = list(range(len(p_1)))
		if len(p_1) == len(p_2) and len(p_1) > 1:
			# get random slice from the first array
			_slice = np.random.permutation(indexes)[:2]
			start, finish = min(_slice), max(_slice)
			p_1 = p_1[start:finish]

			# remove from the second array the values found in the slice of the first array
			
			p_2 = np.array([x for x in p_2 if x in (set(p_2)-set(p_1))])
			if len(p_2) + len(p_1) > len(indexes):
				p_2 = p_2[:len(indexes)-len(p_1)]

			# create the two possible combinations
			p_1, p_2 = np.append(p_1,p_2), np.append(p_2, p_1)
		return [p_1, p_2][np.random.randint(0,2)]