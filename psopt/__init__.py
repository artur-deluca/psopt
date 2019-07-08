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

import logging.config
from os import path
import sys

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log.conf')

logging.config.fileConfig(log_file_path)


logger = logging.getLogger(__name__)

__all__ = ['permutation']

for module in __all__:
    try:
        logger.debug('importing submodule: {}'.format('psopt.{}'.format(module)))
        __import__('psopt.{}'.format(module))
    except Exception as exception:
        logger.error('Error importing {}'.format(module), exc_info=True)
        raise exception
        #sys.exit(0)
