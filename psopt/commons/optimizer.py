import functools
import inspect
import random
import time
import typing
import warnings

import multiprocess
import numpy as np

from psopt.utils import make_logger
from psopt.utils import evaluate_constraints
from psopt.utils import metrics
from psopt.utils import Results
from psopt.utils import get_seeds

Dict = typing.Dict[typing.Text, typing.Any]
List = typing.List[Dict]


class MockPool():
    """
    Class to mock `multiprocessing.Pool` class to avoid
    plickling issues when building documentation but
    also to improve speed
    """

    @staticmethod
    def map(fn, x):
        return list(map(fn, x))

    @staticmethod
    def close():
        pass

    @staticmethod
    def join():
        pass


class Optimizer:
    """Optimizer parent class

    Warning: This class should not be used directly.
    Use derived classes instead"""

    _config = {
        "w": 1,
        "c1": 1,
        "c2": 1,
        "population": 20,
        "max_iter": 300,
        "early_stop": None,
        "threshold": np.inf,
        "penalty": 100,
    }  # type: Dict

    def __init__(
        self,
        obj_func: typing.Callable,
        candidates: list,
        constraints: typing.Optional[typing.Union[Dict, List]] = None,
        **kwargs
    ):

        warnings.filterwarnings("ignore")

        self._candidates = candidates
        self._obj_func = obj_func
        self.n_candidates = len(candidates)
        self.labels = kwargs.get("labels", candidates)  # type: typing.Sequence
        self.metrics = metrics._unpack_metrics(
            kwargs.get("metrics", None)
        )  # type: typing.Optional[typing.Dict[typing.Text, typing.Callable]]
        if isinstance(constraints, dict):
            self.constraints = [constraints]
        else:
            self.constraints = constraints or []

    def maximize(
        self,
        selection_size: typing.Optional[int] = None,
        verbose: typing.Optional[int] = 0,
        **kwargs
    ) -> Results:

        self._set_params(
            selection_size=selection_size, f_min=False, verbose=verbose, **kwargs
        )

        return self._optimize()

    def minimize(
        self,
        selection_size: typing.Optional[int] = None,
        verbose: typing.Optional[int] = 0,
        **kwargs
    ) -> Results:

        self._set_params(
            selection_size=selection_size, f_min=True, verbose=verbose, **kwargs
        )

        return self._optimize()

    def _optimize(self):

        start = time.time()
        if not self._n_jobs or self._n_jobs > 1:
            pool = multiprocess.Pool(self._n_jobs)
        else:
            pool = MockPool()

        # Initialize storage arrays
        self._init_storage_fields()

        # Generate particles
        iteration = 0
        seeds = get_seeds(self.swarm_population)
        self._generate_particles(pool, seeds)

        while iteration < self._max_iter:

            # Add empty placeholders for the calculation by copying templates
            # already defined at `init_storage_fields`
            self._particles.append(self._template_position.copy())
            self._particles_best.append(self._template_position.copy())
            self._global_best.append(self._template_global.copy())

            # In case of a dynamic inertia configuration, function calculates
            # the according weight for each iteration
            self._w = self._update_w(iteration)

            # Evaluate the latest generated particles
            # according to the objective function and the compliance
            # with existing constraints as well
            self._evaluate_particles(pool)

            # Identifies best values for the whole swarm and for each particle
            exit_flag = self._update_best()

            # Logging iteration
            iteration_best_index = np.argmax(self._particles[-2]["value"])
            message = "Iteration {}:\n".format(iteration)
            metric_results = {
                "global_best": self._m * self._global_best[-2]["value"],
                "iteration_best": self._m * self._particles[-2]["value"][iteration_best_index]
            }

            # Log metric results
            metric_results = {**metric_results, **self._calculate_metrics()}

            message += "".join(
                [
                    "   {}: {:.3f}".format(key, value)
                    for key, value in metric_results.items()
                ]
            )
            self._logger.info(message)
            self._logger.write_metrics(metric_results)

            if exit_flag:
                break

            seeds = get_seeds(self.swarm_population)
            self._update_components(pool, seeds)

            # Remove unnecessary and used storage arrays
            # TODO: Record all the iterations for future debugging purposes
            if not self._record and iteration > 1:
                self._particles.pop(0)
                self._particles_best.pop(0)
                self._global_best.pop(0)

            # Stopping criteria
            if len(np.unique(self._particles[-1]["position"], axis=0)) == 1:
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
        solution.meta = self.metadata
        solution.results = {
            "solution": [int(x) for x in self._global_best[-2]["position"]],
            "value": float(self._m * self._global_best[-2]["value"]),
            "elapsed_time": float("{:.3f}".format(time.time() - start)),
            "exit_status": exit_flag,
            "iterations": iteration,
        }

        constraint_check = evaluate_constraints(
            self.constraints, self._get_particle(solution.solution)
        )

        solution.results["solution"] = self._get_labels(solution.solution)

        if constraint_check > 0:
            solution.results["feasible"] = False
            self._logger.warn(
                "The algorithm was unable to find a feasible"
                "solution with the given parameters"
            )
        else:
            solution.results["feasible"] = True

        solution.load_history(self._logger.file_path, delete=True)

        self._logger.info("Elapsed time {}".format(solution.results["elapsed_time"]))
        self._logger.info("{} iterations".format(iteration))
        self._logger.info("Best selection: {}".format(solution.solution))
        self._logger.info("Best evaluation: {}".format(solution.value))

        return solution

    def _evaluate_particles(self, pool):
        params = [
            {
                "particle": self._get_particle(particle),
                "m": self._m,
                "obj_fn": self._obj_func,
                "constraints": self.constraints,
                "penalty": self._penalty,
            }
            for particle in self._particles[-2]["position"]
        ]

        self._particles[-2]["value"] = np.array(
            pool.map(self._calculate_obj_fn, params)
        )

    @staticmethod
    def _calculate_obj_fn(
        params: typing.Dict[str, typing.Any]
    ) -> typing.Union[float, int]:

        # Evaluate particle on the objective function
        evaluation = params["m"] * params["obj_fn"](params["particle"])

        # Add potential penalties caused by constraints' violations
        constraint_factor = evaluate_constraints(
            params["constraints"], params["particle"]
        )
        evaluation += params["m"] * params["penalty"] * constraint_factor

        return evaluation

    def _update_best(self):

        # For early stopping use
        last_best_position = list(self._global_best[-2]["position"])

        # Temporarily set the best particle values and
        # position as the most recent iteration
        self._particles_best[-2] = self._particles[-2].copy()

        # Get the last 3 particle best values for each particle
        all_values = np.array([i["value"] for i in self._particles_best[-4:-1]])

        # Assign the current particle best value as the
        # maximum of the previous selection
        self._particles_best[-2]["value"] = all_values.max(axis=0)

        # Assign the corresponding position accordingly
        self._particles_best[-2]["position"] = [
            self._particles_best[x]["position"][i]
            for i, x in enumerate(all_values.argmax(axis=0))
        ]

        # Set the current and next global best values accordingly
        g_best = self._particles_best[-2]["value"].max()
        self._global_best[-2]["value"] = g_best
        self._global_best[-1]["value"] = g_best

        # Set the current global best position
        gbest_index = self._particles_best[-2]["value"].argmax()
        gbest_position = self._particles_best[-2]["position"][gbest_index]
        self._global_best[-2]["position"] = gbest_position

        # If the best position has been changed
        if last_best_position != list(gbest_position):
            # Clear counter since new global best was found
            self._early_stop_counter = 0
            if self._global_best[-2]["value"] >= self._threshold:
                return 2  # Set exit flag no.2
        else:
            self._early_stop_counter += 1
            if self._early_stop_counter >= self._early_stop:
                return 1  # Set exit flag no.1

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
        for field in __class__._config:
            if field in kwargs.keys():
                setattr(self, "_{}".format(field), kwargs[field])
            else:
                setattr(self, "_{}".format(field), __class__._config[field])

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
            w_sta, w_fin = tuple(self._w)

            self._update_w = lambda i: (w_fin - (w_fin - w_sta)) * (
                i / self._max_iter
            )  # noqa: E731

        except TypeError:
            self._update_w = lambda i: self._w  # noqa: E731

        self._n_jobs = kwargs.get("n_jobs", 1)
        if self._n_jobs == -1:
            self._n_jobs = None
        elif self._n_jobs == -2:
            self._n_jobs = multiprocess.cpu_count() - 1

        self._seed = kwargs.get("seed", None)
        random.seed(self._seed)

    @property
    def metadata(self):
        try:
            metadata = {
                "c1": self._c1,
                "c2": self._c2,
                "w": self._w,
                "population": self.swarm_population,
                "max iterations": self._max_iter,
                "early_stop": self._early_stop,
                "threshold": self._threshold,
                "constraint_penalty": self._penalty,
            }

            return metadata

        except AttributeError:
            warnings.warn(
                "Metadata not set yet. Please run `minimize` or "
                "`maximize` to generate metadata",
                Warning,
            )
            return None

    def _exit(self, flag: int):

        flag = flag or 0

        exit_flag = {
            0: "Algortihm reached the maximum limit"
            " of {} iterations".format(self._max_iter),
            1: "Algorithm has not improved for"
            " {} consecutive iterations".format(self._early_stop),
            2: "Algorithm has reached the value "
            "threshold of {}".format(self._m * self._threshold),
            3: "Particles converged to a single solution",
        }

        self._logger.info("\nIteration completed\n" "==========================")

        self._logger.info("Exit code {}: {}".format(flag, exit_flag[flag]))

    def _calculate_metrics(self):

        metric_results = dict()  # set dict to store the results
        positions = [self._get_particle(i) for i in self._particles[-2]["position"]]
        for name, func in self.metrics.items():

            # if the number of parameters is equal 2, partially complete it
            # with the global best at current iteration
            number_of_param = len(inspect.signature(func).parameters)
            if number_of_param == 2:
                func = functools.partial(func, self._get_particle(self._global_best[-2]["position"]))

            # calculate metrics
            metric_results[name] = func(positions)

        return metric_results

    def _init_storage_fields(self):
        self._template_position = {
            "position": [[] for _ in range(self.swarm_population)],
            "value": [-np.inf for _ in range(self.swarm_population)],
        }

        self._template_global = {"position": [], "value": -np.inf}

        # particles[iteration][position or value][particle]
        self._particles = [self._template_position.copy()]

        # particles_best[iteration][position or value][particle]
        self._particles_best = [self._template_position.copy()]

        # global_best[iteration][position or value]
        self._global_best = [self._template_global.copy()]

    
    def _generate_particles(self, pool, seeds):
        pass

    
    def _update_components(self, pool, seeds):
        pass

    def _get_particle(self, position):
        pass

    def _get_labels(self, position):
        pass
