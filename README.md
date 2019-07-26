# PSOpt

A particle swarm optimizer for combinatorial optimization

# How to use
```
from psopt.permutation import Permutation

# define an objective function to optimize
def obj_func(x):
    return sum([a / (i + 1) for i, a in enumerate(x)])

# list of possible candidates
candidates = list(range(1, 11))

# instantiate the optimizer
opt = Permutation(obj_func, candidates, metrics="l2")

# minimize the obj function
result = opt.minimize(selection_size=5, verbose=1, threshold=threshold, population=20)

# visualize the progress
result.history.plot("l2")
result.history.plot("global_best")
```