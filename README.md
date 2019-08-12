<p><img width=400 src="/docs/images/psopt.svg"></p>

[![Build Status](https://travis-ci.com/artur-deluca/psopt.svg?branch=master)](https://travis-ci.com/artur-deluca/psopt)
[![codecov](https://codecov.io/gh/artur-deluca/psopt/branch/master/graph/badge.svg)](https://codecov.io/gh/artur-deluca/psopt)
[![Maintainability](https://api.codeclimate.com/v1/badges/e969d457f95dca89cb31/maintainability)](https://codeclimate.com/github/artur-deluca/psopt/maintainability)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/71b0d894f71f4c7c9f14409d14b11856)](https://www.codacy.com/app/artur-deluca/psopt?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=artur-deluca/psopt&amp;utm_campaign=Badge_Grade)

A particle swarm optimizer for combinatorial optimization

## Project Information

`psopt` is released under the [MIT](https://choosealicense.com/licenses/mit/),
its documentation lives at [Read the Docs](https://psopt.readthedocs.io/en/latest/),
the code on [GitHub](https://github.com/artur-deluca/psopt),
and the latest release on [PyPI](https://pypi.org/project/psopt/).

If you'd like to contribute to `psopt` you're most welcome. We've created a [little guide](CONTRIBUTING.md) to get you started!

## How to use
```python
from psopt import Permutation

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
#result.history.plot("l2")
result.history.plot("global_best")
```

<p>
  <img width="400" height="300" src="/docs/images/global_best.svg">
</p>

## Installation

### Python version support

Officially Python 3.6 and above

### Installing from PyPI

PSOpt can be installed via pip from [PyPI](https://pypi.org/project/psopt)

```  
pip install psopt
```

### Installing from source

Clone the repository via:

```
git clone https://github.com/artur-deluca/psopt/ --depth=1
```

Activate your virtual environment and run:

```
python setup.py install
# or alternatively
pip install -e .
```

If you wish to install the development dependencies, run:

```
python setup.py build
# or alternatively
pip install -e.[all]
```

### Running the test suite

To run the tests written for psopt, make sure you have `pytest` installed in your venv. 
Additionally, if you wish to run coverage analysis as well, make sure to have `pytest-cov` installed as well.

```
# to simply execute the tests run:
pytest
# to run coverage as well run:
pytest --cov=psopt
# or alternatively:
make test
```

