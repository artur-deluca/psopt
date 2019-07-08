"""
Combination sub-module
=====

A particle swarm optimization algorithms used to find the best combination of values


Available
---------------------
CombinationOptimizer

"""

import logging.config
from os import path

log_file_path = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'log.conf')
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)


logger = logging.getLogger(__name__)
from .optimizer import *
