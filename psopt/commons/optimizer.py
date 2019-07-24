import functools
import inspect
import multiprocess
import time
import typing
import warnings

import numpy as np

from psopt.utils import make_logger
from psopt.utils import evaluate_constraints
from psopt.utils import metrics
from psopt.utils import Results

Dict = typing.Dict[str, typing.Any]
List = typing.List[Dict]


class Optimizer:
    """Optimizer parent class"""

    config = {
        "w": 1,
        "c1": 1,
        "c2": 1,
        "population": 20,
        "max_iter": 300,
        "early_stop": None,
        "threshold": np.inf,
        "penalty": 100,
    }

    def __init__(self,
                 obj_func: typing.Callable,
                 candidates: list,
                 constraints: typing.Optional[typing.Union[Dict, List]],
                 **kwargs):

        warnings.filterwarnings("ignore")

        self._candidates = candidates
        self._obj_func = obj_func
        self.n_candidates = len(candidates)
        self.labels = kwargs.get("labels", candidates)  # type: typing.Sequence
        self.metrics = metrics.unpack_metrics(kwargs.get("metrics"))  # type: typing.Optional[typing.Dict[str, typing.Callable]]

        if isinstance(constraints, dict):
            self.constraints = [constraints]
        else:
            self.constraints = constraints or []

    def maximize(self,
                 selection_size: typing.Optional[int],
                 verbose: typing.Optional[int] = 0,
                 **kwargs) -> Results:
        """Seeks the solution that yields the maximum objective function value

        Args:
            selection_size: int, number of candidates to compose a solution.

            verbose: int, controls the verbosity while optimizing
                0: Display nothing (default);
                1: Display statuses on console;
                2: Display statuses on console and store them in .logs.

            w [inertia]: float or sequence (float, float)
                It controls the contribution of the previous movement.
                If a single value is provided, w is fixed, otherwise it linearly alters from min to max within the sequence provided.

            c1 [self-confidence]: float
                It controls the contribution derived by the difference between a particle's current position and it's best position found so far.

            c2 [swarm-confidence]: float
                It controls the contribution derived by the difference between a particle's current position and the swarm's best position found so far.

            population: float or int
                Factor to cover the search space (e.g. 0.5 would generate a number of particles of half the search space).
                If `population` is greater than one, the population size will have its value assigned.

            max_iter: int, default 300
                Maximum possible number of iterations.

            early_stop: int, default max_iter
                Maximum number of consecutive iterations with no improvement that the algorithm accepts without stopping.

        Returns:
            a Result object containing the solution, metadata and stored metrics history
        """

        self._set_params(selection_size=selection_size, f_min=False, verbose=verbose, **kwargs)

        return self._optimize()

    def minimize(self,
                 selection_size: typing.Optional[int],
                 verbose: typing.Optional[int] = 0,
                 **kwargs) -> Results:
        """Seeks the solution that yields the minimum objective function value

        Args:
            selection_size: int, number of candidates to compose a solution.

            verbose: int, controls the verbosity while optimizing
                0: Display nothing (default);
                1: Display statuses on console;
                2: Display statuses on console and store them in .logs.

            w [inertia]: float or sequence (float, float)
                It controls the contribution of the previous movement.
                If a single value is provided, w is fixed, otherwise it linearly alters from min to max within the sequence provided.

            c1 [self-confidence]: float
                It controls the contribution derived by the difference between a particle's current position and it's best position found so far.

            c2 [swarm-confidence]: float
                It controls the contribution derived by the difference between a particle's current position and the swarm's best position found so far.

            population: float or int
                Factor to cover the search space (e.g. 0.5 would generate a number of particles of half the search space).
                If `population` is greater than one, the population size will have its value assigned.

            max_iter: int, default 300
                Maximum possible number of iterations.

            early_stop: int, default max_iter
                Maximum number of consecutive iterations with no improvement that the algorithm accepts without stopping.

        Returns:
            a Result object containing the solution, metadata and stored metrics history
        """

        self._set_params(selection_size=selection_size, f_min=True, verbose=verbose, **kwargs)

        return self._optimize()

    def _optimize(self):

        start = time.time()
        pool = multiprocess.Pool()

        # Initialize storage arrays
        self._init_particles()

        # Generate particles
        iteration = 0
        self._particles[-1]["position"] = pool.map(self._generate_particles, range(self.swarm_population))

        while(iteration < self._max_iter):
            self._particles.append(self._template_position.copy())
            self._particles_best.append(self._template_position.copy())
            self._global_best.append(self._template_global.copy())
            self._w = self._update_w(iteration)

            self._particles[-2]["value"] = np.array(pool.map(self._multi_obj_func, range(self.swarm_population)))

            exit_flag = self._update_best()
            if exit_flag: break

            # Logging iteration
            message = "Iteration {}:\n".format(iteration)

            measure_results = {
                "global_best": self._m * self._global_best[-2]["value"],
                "iteration_best": self._m * max(self._particles[-2]["value"])
            }

            # Log metric results
            measure_results = {**measure_results, **self._calculate_measures(pool=pool)}
            message += "".join(
                [
                    "   {}: {:.3f}".format(key.replace("_", " "), value)
                    for key, value in measure_results.items()
                ]
            )
            self._logger.info(message)
            self._logger.write_metrics(measure_results)

            self._update_particles(pool=pool)

            # Record all the iterations for future debugging purposes
            if not self._record and iteration > 1:
                self._particles.pop(0)
                self._particles_best.pop(0)
                self._global_best.pop(0)

            # Stopping criteria
            unique = np.unique(self._particles[-1]["position"])

            if len(unique) == self.swarm_population - 1:
                exit_flag = 3
                break
            else:
                iteration += 1
                continue
            break  # Break the loop whenever the if clause also breaks

        # Close the pool of workers
        pool.close()
        pool.join()

        # Output the exit flag
        self._exit(exit_flag)

        # Store the results
        solution = Results()
        solution.meta = self.get_metadata()
        solution.meta["results"] = {
            "solution_index": list(map(int, self._global_best[-2]["position"])),
            "solution_value": float(self._m * self._global_best[-2]["value"]),
            "elapsed_time": float("{:.3f}".format(time.time() - start)),
            "exit_status": exit_flag,
            "iterations": iteration
        }

        results = solution.meta["results"]  # For the sake of concision
        if evaluate_constraints(self.constraints, self._get_particle(results["solution_index"])) > 0:
            results["feasible"] = False
            self._logger.warn("The algorithm was unable to find a feasible solution with the given parameters")
        else:
            results["feasible"] = True

        self._logger.write_meta(solution.meta)

        solution.load_history(self._logger.file_path)
        solution.solution = self._get_labels(results["solution_index"])

        self._logger.info("Elapsed time {}".format(results["elapsed_time"]))
        self._logger.info("{} iterations".format(iteration))
        self._logger.info("Best selection: {}".format(solution.solution))
        self._logger.info("Best evaluation: {}".format(results["solution_value"]))

        return solution

    def _multi_obj_func(self, i: int) -> typing.Union[float, int]:

        # Get real values of particle
        particle = self._get_particle(self._particles[-2]["position"][i])

        # Evaluate particle on the objective function
        evaluation = self._m * self._obj_func(particle)

        # Add potential penalties caused by constraints' violations
        evaluation += self._penalty * (self._m * evaluate_constraints(self.constraints, particle))

        return evaluation

    def _update_best(self):

        # For early stopping use
        last_best_position = list(self._global_best[-2]["position"])

        # Temporarily set the best particle values and position as the most recent iteration
        self._particles_best[-2] = self._particles[-2]

        # Get the last 3 particle best values for each particle
        all_values = np.array([i["value"] for i in self._particles_best[-4:-1]])

        # Assign the current particle best value as the maximum of the previous selection
        self._particles_best[-2]["value"] = all_values.max(axis=0)

        # Assign the corresponding position accordingly
        self._particles_best[-2]["position"] = [self._particles_best[x]["position"][i] for i, x in enumerate(all_values.argmax(axis=0))]

        # Set the current and next global best values accordingly
        self._global_best[-2]["value"] = self._particles_best[-2]["value"].max()
        self._global_best[-2]["position"] = self._particles_best[-2]["position"][self._particles_best[-2]["value"].argmax()]
        self._global_best[-1]["value"] = self._global_best[-2]["value"]

        # If the best position has been changed
        if (last_best_position != list(self._global_best[-2]["position"])):
            self._early_stop_counter = 0  # Clear counter since new global best was found
            if self._global_best[-2]["value"] >= self._threshold: return 2  # Set exit flag no.2
        else:
            self._early_stop_counter += 1
            if self._early_stop_counter >= self._early_stop: return 1  # Set exit flag no.1

        return None

    def _set_params(self, selection_size, f_min, verbose, **kwargs):

        # Set optimizer logger
        self._logger = make_logger(__name__, verbose=verbose)

        # Record all iterations
        self._record = kwargs.get("record", False)  # type: bool

        # Transform the maximization problem into a minimization
        self._m = 1 - 2 * f_min

        # Set the selection size
        self.selection_size = selection_size or self.n_candidates

        # Set the swarm and optimization parameters
        for field in __class__.config:
            if field in kwargs.keys():
                setattr(self, "_{}".format(field), kwargs[field])
            else:
                setattr(self, "_{}".format(field), __class__.config[field])

        # Configure acceptable threshold
        if self._threshold != np.inf:
            self._threshold *= self._m

        # Configure early stopping if is None
        self._early_stop = self._early_stop or self._max_iter

        # Set the number of particles (population)
        if self._population > 1:
            self.swarm_population = int(self._population)
        else:
            self.swarm_population = int(self.n_candidates * self._population)

        # Set the inertia function
        try:
            w_start, w_finish = tuple(self._w)
            self._update_w = lambda i: (w_finish - ((w_finish - w_start) * (i / self._max_iter)))  # noqa: E731

        except TypeError:
            self._update_w = lambda i: self._w  # noqa: E731

    def get_metadata(self):
        metadata = {
            "initial_config": {
                "c1": self._c1,
                "c2": self._c2,
                "w": self._w,
                "population": self.swarm_population,
                "max iterations": self._max_iter,
                "early_stop": self._early_stop,
                "threshold": self._threshold,
                "constraint_penalty": self._penalty,
            }
        }

        return metadata

    def _exit(self, flag: int):

        flag = flag or 0

        exit_flag = {
            0: "Algortihm reached the maximum limit of {} iterations".format(self._max_iter),
            1: "Algorithm has not improved for {} consecutive iterations".format(self._early_stop),
            2: "Algorithm has reached the value threshold of {}".format(self._m * self._threshold),
            3: "Particles converged to a single solution"
        }

        print()
        self._logger.info("Iteration completed\n"
                          "==========================")

        self._logger.info("Exit code {}: {}".format(flag, exit_flag[flag]))

    def _calculate_measures(self, pool):

        measure_results = dict()  # set dict to store the results
        for name, func in self.metrics.items():

            # if the number of parameters is equal 2, partially complete it with the global best at current iteration
            number_of_param = len(inspect.signature(func).parameters)
            if number_of_param == 2:
                func = functools.partial(func, self._global_best[-2]["position"])

            # calculate measures
            measure_results[name] = np.mean(pool.map(func, self._particles[-2]["position"]))

        return measure_results

    def _init_particles(self):
        pass

    def _get_particle(self, position):
        pass

    def _get_labels(self, position):
        pass

    def _update_particles(self, **kwargs):
        pass
