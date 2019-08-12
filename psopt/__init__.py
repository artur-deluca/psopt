"""
PSOpt
=====

A particle swarm optimization tool for general purpose

How to use the documentation
----------------------------
Documentation is available in docstrings provided with the code.

Available
---------------------
Permutation
    Optimizer to find the best permutation of possibilities

Combination
    Optimizer to find the best combination of possibilities

"""
__version__ = "0.1.0"

__title__ = "PSOpt"
__description__ = "Particle swarm optimization in Python"
__url__ = "https://psopt.readthedocs.io"

__author__ = "Artur Back de Luca"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2019 Artur Back de Luca"


from psopt.permutation import Permutation
from psopt.combination import Combination
