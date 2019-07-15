"""
PSOpt
=====

A particle swarm optimization tool for general purpose

How to use the documentation
----------------------------
Documentation is available in docstrings provided
with the code.

Available
---------------------
permutation
    sub-module dedicated to find the best permutation of data

combination
    sub-module dedicated to find the best combination of data

"""

import psopt.permutation
import psopt.combination
from psopt.utils import make_logger


__all__ = ['make_logger']