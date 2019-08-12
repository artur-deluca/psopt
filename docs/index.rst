*****************************************
PSOpt: a particle swarm optimization tool
*****************************************

|psopt|_ is an open source package for general use based on Particle swarm optimization (PSO).
PSO is a population based stochastic optimization technique developed by Dr. Eberhart and Dr. Kennedy  in 1995, inspired by social behavior of bird flocking or fish schooling.

PSO shares many similarities with evolutionary computation techniques such as Genetic Algorithms (GA). The system is initialized with a population of random solutions and searches for optima by updating generations. However, unlike GA, PSO has no evolution operators such as crossover and mutation. In PSO, the potential solutions, called particles, fly through the problem space by following the current optimum particles. [1]


``psopt`` is released under the `MIT <https://choosealicense.com/licenses/mit/>`_ license,
its documentation lives at `Read the Docs <https://psopt.readthedocs.io/en/latest/>`_,
the code on `GitHub <https://github.com/artur-deluca/psopt>`_,
and the latest release on `PyPI <https://pypi.org/project/psopt/>`_.


.. note:: Currently PSOpt only supports combinatorial optimization.

[1] Hu, X. (2006). Particle Swarm Optimization <http://www.swarmintelligence.org/>

Overview
===============

.. toctree::

    source/overview/install
    source/overview/contributing


API Reference
-------------
.. autosummary::
   :toctree:

   psopt.Combination
   psopt.Permutation
   psopt.utils.Results
   psopt.utils.metrics


.. |psopt| replace:: ``psopt``
.. _psopt: https://github.com/artur-deluca/psopt/