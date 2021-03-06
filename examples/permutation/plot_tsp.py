"""
=================================
Travelling salesman problem (TSP)
=================================


.. note::example taken from `Google OR-Tools <https://developers.google.com/optimization/routing/tsp>`_

The travelling salesman problem (TSP) concerns on finding the shortest
possible route linking a given list of cities and their distances.

The TSP is one of the most famous problems in computer science and it belongs to the class of NP-Complete problems.
The Travelling salesman problem have many important applications, from route planning to the manufacture of microchips.
But, in essence, it can be formulated as to find the minimum path that links each node in a weighted graph.

.. image:: /images/tsp.svg

Using ``psopt.Permutation`` we can generate many instances that use different paths. Then, those paths that have yielded the best results guide the other particles on finding better results throughout iterations. Below we see the progress of the algorithm:
"""

import functools
from psopt import Permutation


def main():

    def get_route_cost(x, initial_cost, distance_matrix):

        # define initial route cost
        route_cost = initial_cost

        for i in range(len(x) - 1):
            route_cost += distance_matrix[x[i]][x[i + 1]]

        return route_cost

    # define distances between 13 cities
    distance_matrix = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]

    obj_func = functools.partial(
        get_route_cost, initial_cost=0, distance_matrix=distance_matrix
    )
    candidates = list(range(len(distance_matrix)))

    # instantiate the optimizer
    opt = Permutation(obj_func, candidates)

    # minimize the obj function
    solution = opt.minimize(
        verbose=1, max_iter=50, population=15, early_stop=25, seed=0
    )

    solution.history.plot("global_best")


if __name__ == "__main__":
    main()
