import random
from psopt.permutation import Permutation as optim

if __name__ == "__main__":

    # define objective function: f([a, b, c, ...]) = a/1 + b/2 + c/3 + ...
    def obj_func(x):
        return sum([a / (i + 1) for i, a in enumerate(x)])

    # list of possible candidates
    candidates = list(range(1, 26))
    random.shuffle(candidates)

    # instantiate the optimizer
    opt = optim(obj_func, candidates)

    # define a threshold of acceptance for early convergence
    threshold = obj_func(sorted(candidates))

    # minimize the obj function
    opt.minimize(verbose=True, population=20)
