import random
from psopt.permutation import PermutationOptimizer as optim

if __name__ == '__main__':

    # define objective function: f([a, b, c, ...]) = a/1 + b/2 + c/3 + ...
    def obj_func(x):
        return sum([a / (i + 1) for i, a in enumerate(x)])

    # list of possible candidates
    candidates = list(range(1, 11))
    random.shuffle(candidates)

    selection_size = 5

    # constraint: sum of values cannot be greater than 16
    constraint = {"fn": sum, "type": ">", "value": sum(sorted(candidates)[:selection_size]) + 1}

    # instantiate the optimizer
    opt = optim(obj_func, candidates, constraints=constraint)

    # define a threshold of acceptance for early convergence
    threshold = obj_func(sorted(candidates)[:selection_size])

    # minimize the obj function
    opt.minimize(selection_size=selection_size, verbose=True, threshold=threshold, population=20)
