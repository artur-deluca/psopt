import logging.config
import numpy as np
import os
import pathos.multiprocessing as multiprocessing
import time
from psopt.utils import evaluate_constraints
import warnings



class PermutationOptimizer:
	"""Particle swarm permutation optimizer to find the best permutation of candidates
	
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
		>>> def obj_func(x):
		...		value = 0
		...		for i in range(len(x)):
		...			value += x[i]/(i+1)
		...		return value

		>>> # sum of items in solution can be no greater than 22
		>>> def constraint(x):
		...		clause = 0
		...		for i in x:
		...			clause += candidates[x]
		...		return clause <= 22
		
		>>> opt = PermutationOptimizer(obj_func, candidates, constraint)
		>>> opt.minimize(selection_size=5)

	"""

	config = {
			"w": 0.7,
			"c1": 1.4,
			"c2": 1.4,
			"population": 0.4,
			"max_iter": 300,
			"early_stop": None,
			"threshold": np.inf,
			"epsilon": 0,
			"penalty": 1000
	}


	def __init__(self, obj_func, candidates, constraints=[]):

		self._candidates = candidates
		self._obj_func = obj_func
		self.n_candidates = len(candidates)
		warnings.filterwarnings("ignore")
		
		if type(constraints) is dict:
			self.constraints = [constraints]
		else:
			self.constraints = constraints

		log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log.conf')
		logging.config.fileConfig(log_file_path, disable_existing_loggers=False)

		self._logger = logging.getLogger()
		self._logger.name = __name__

	def maximize(self, selection_size=None, verbose=False, **kwargs):
		"""Seeks the best permutation of candidates that yields the maximum objective function value

		Parameters
    	----------
			selection_size: int, optional
				The number of candidates to compose a solution. If not declared, the total number of candidates will be used as the selection size
			
			verbose: bool, default False
		
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
		self._set_params(selection_size=selection_size, f_min=False, verbose=verbose, **kwargs)

		return self._optimize()
	
	def minimize(self, selection_size=None, verbose=False, **kwargs):
		"""Seeks the best permutation of candidates that yields the minimum objective function value
		
		Parameters
    	----------
			selection_size: int, optional
				The number of candidates to compose a solution. If not declared, the total number of candidates will be used as the selection size
			
			verbose: bool, default False
		
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
		self._set_params(selection_size=selection_size, f_min=True, verbose=verbose, **kwargs)
		
		return self._optimize()
	
	def _optimize(self):
		
		start = time.time()

		# set default exit flag
		exit_flag = 0

		# create pool for parallel processing
		pool = multiprocessing.Pool()
		
		# initialize "storage" arrays
		self._init_particles(self.n_candidates)

		# initialize particles
		self._generate_particles(constraints=self.constraints, pool_size=self.n_candidates)
		
		# optimizing
		iteration = 0
		while(1 and iteration < self._max_iter - 1):
			
			self._particles.append(self._template_position)
			self._particles_best.append(self._template_position)
			self._global_best.append(self._template_global)

			args_ = list(
				zip(
					list(range(self.swarm_population)),
					[iteration] * self.swarm_population
				)
			)

			results = pool.starmap(self._parallelize_func, args_)
			results = list(map(list, zip(*results)))

			self._particles[iteration]["value"] = np.array(results[0])
			self._particles_best[iteration]["value"] = np.array(results[1])
			self._particles_best[iteration]["position"] = results[2]

			# for early stopping use
			last_best = self._global_best[iteration]["position"]
			
			# update Global Best
			if (self._global_best[iteration]["value"] < max(self._particles_best[iteration]["value"])):
				
				early_stop_counter = 0 # clear counter since new global best was found

				self._global_best[iteration]["value"] = max(self._particles_best[iteration]["value"])
				self._global_best[iteration + 1]["value"] = self._global_best[iteration]["value"]
				self._global_best[iteration]["position"] = self._particles_best[iteration]["position"][self._particles_best[iteration]["value"].argmax()]

				if self._global_best[iteration]["value"] > self._threshold:
					exit_flag = 2
					break

			else:
				self._global_best[iteration + 1]["value"] = self._global_best[iteration]["value"]
				self._global_best[iteration]["position"] = self._global_best[iteration - 1]["position"]
				
				# it may have the same value and not the same position
				if (last_best == self._global_best[iteration]["position"]).all():
					early_stop_counter += 1
					if early_stop_counter >= self._early_stop:
						exit_flag = 1
						break
				else:
					early_stop_counter = 0
					
			self._logger.debug('\rIteration %d gbest = %f and iteration best = %f' % (iteration, self._m * self._global_best[iteration]["value"], self._m * max(self._particles[iteration]["value"])))
			
			# velocities Update
			self._velocities = self.w(iteration) * self._velocities

			Position = np.array(self._particles[iteration]["position"])
			particle_best_comp = self._particles_best[iteration]["position"] - Position

			self._velocities += self._c1 * np.random.random(particle_best_comp.shape) * particle_best_comp

			global_best_comp = np.tile(self._global_best[iteration]["position"], (self.swarm_population, 1)) - Position

			self._velocities += self._c2 * np.random.random() * global_best_comp

			# velocity clamping
			self._velocities[self._velocities > self.n_candidates/2] = self.n_candidates/2


			Position = Position + np.round(self._velocities)

			# Boundary Checking for Position
			Position[Position > self.n_candidates - 1] = self.n_candidates - 1
			Position[Position < 0] = 0

			for i in range(0, self.swarm_population):
				if (len(np.unique(Position[i])) != self.selection_size):
					tempVar = np.random.permutation(self.selection_size)
					Position[i] = tempVar[0:self.selection_size]

			for i in range(self.swarm_population):
				self._particles[iteration + 1]["position"][i] = Position[i].astype(int)

			# stop criteria

			unique = np.unique(self._particles[iteration + 1]['position'])

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
		solution = [self._candidates[x] for x in self._global_best[iteration]["position"]]
		solution_value = self._m * self._global_best[iteration]["value"]
		elapsed_time = time.time() - start  # noqa: F841

		self._exit(exit_flag)
		
		if evaluate_constraints(self.constraints, [self._candidates[x] for x in self._global_best[iteration]["position"]]) > 0:
			self._logger.info("The algorithm was unable to find a feasible solution")

		self._logger.info("Elapsed time {}".format(elapsed_time))
		self._logger.info("{} iterations".format(iteration))
		self._logger.info("Best selection: {}".format(solution))
		self._logger.info("Best evaluation: {}".format(solution_value))

		return solution

	def _parallelize_func(self, i, iteration):
		
		particle = [self._candidates[x] for x in self._particles[iteration]["position"][i]]
		
		evaluation = self._m  * self._obj_func(particle)

		evaluation += self._penalty * (self._m  * evaluate_constraints(self.constraints, particle))

		evaluations = np.array([x["value"][i] for x in self._particles[:iteration + 1]])

		if evaluation > max(evaluations):
			best = evaluation
			best_selection = self._particles[iteration]["position"][i]
		else:
			best = max(evaluations)
			best_selection = self._particles[evaluations.argmax()]["position"][i]

		return [evaluation, best, best_selection]

	def _init_particles(self, pool_size):
		
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

	def _generate_particles(self, constraints, pool_size, iteration=0):
		
		for i in range(0, self.swarm_population):

			candidates = np.random.permutation(np.arange(pool_size))
			candidates = candidates[0:self.selection_size]

			self._particles[iteration]["position"][i] = candidates

	def _set_params(self, selection_size, f_min, verbose, **kwargs):
		
		# quick fix
		# just a reminder that this sets the logging level for all instances of root as well
		if verbose:
			self._logger.handlers[0].setLevel(logging.DEBUG)
		else:
			self._logger.handlers[0].setLevel(logging.WARNING)

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
			self.w = lambda i: (w_max - ((w_max - w_min) * (i / self._max_iter)))  # noqa: E731
		
		except TypeError:
			w_input = self._w
			self.w = lambda i: w_input  # noqa: E731

	def _exit(self, flag):
		
		exit_flag = {
			0: "Algortihm reached the maximum limit of {} iterations".format(self._max_iter),
			1: "Algorithm has not improved for {} consecutive iterations".format(self._early_stop),
			2: "Algorithm has reached the value threshold of {}".format(self._threshold),
			3: "Particles converged to a single solution"
		}

		self._logger.info("Iteration completed")
		self._logger.info("Exit code {}: {}".format(flag, exit_flag[flag]))