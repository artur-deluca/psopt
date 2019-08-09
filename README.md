<p>
  <img width=400 src="/docs/images/psopt.svg">
</p>

[![Build Status](https://travis-ci.com/artur-deluca/psopt.svg?branch=master)](https://travis-ci.com/artur-deluca/psopt)
[![codecov](https://codecov.io/gh/artur-deluca/psopt/branch/master/graph/badge.svg)](https://codecov.io/gh/artur-deluca/psopt)
[![Maintainability](https://api.codeclimate.com/v1/badges/e969d457f95dca89cb31/maintainability)](https://codeclimate.com/github/artur-deluca/psopt/maintainability)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/71b0d894f71f4c7c9f14409d14b11856)](https://www.codacy.com/app/artur-deluca/psopt?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=artur-deluca/psopt&amp;utm_campaign=Badge_Grade)

A particle swarm optimizer for combinatorial optimization

## How to use
```python
from psopt.permutation import Permutation

# define an objective function to optimize
def obj_func(x):
    return sum([a / (i + 1) for i, a in enumerate(x)])

# list of possible candidates
candidates = list(range(1, 11))

# instantiate the optimizer
opt = Permutation(obj_func, candidates, metrics="l2")

# minimize the obj function
result = opt.minimize(selection_size=5, verbose=1, threshold=5, population=20)

# visualize the progress
result.history.plot("l2")
result.history.plot("global_best")
```
