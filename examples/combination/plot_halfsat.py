"""
===================================================
Half-SAT problem
===================================================

The Half-SAT problem can be defined as:

.. math::

    \\text{HALF-SAT} = \\{ \\varphi \\mid \\varphi \\text{ is a formula which is satisfied by half of all assignments }\\}

Or, in other words, considering a CNF formula with :math:`2n` variables, an there's a satisfying assignment in which :math:`n` variables are ``True`` and :math:`n` variables are ``False``.
The Half-SAT problem is known to be `NP-Hard <https://en.wikipedia.org/wiki/NP-hardness>`_, so it cannot be solved in polynomial time.
In these situations the Particle Swarm Algorithm can sometimes reach optimal results in an feasible amount of time

"""

import random
from psopt import Combination


def generate_cnf(n_vars, clauses, k=3, seed=None):
    """
    This function generates a random CNF formula for the Half-SAT problem

    Args:
        n_vars (int): the number of variables in the formula

        clauses (int): the number of clauses that compose the formula

        k (int, optional): the number of variables in each clause. Defaults to 3

        seed (int, optional): Seed the function generator

    Returns:
        cnf (callable): a function in CNF for the Half-SAT problem
    """
    # seed value for testing purposes
    random.seed(seed)
    # start expression
    cnf = str()
    # iterate clauses
    for _ in range(clauses):
        clause = str()
        # get variables
        variables = random.sample(list(range(n_vars)), n_vars)[:k]
        # build clause
        clause += random.choice(["", "not "]) + "x[{}]".format(variables.pop(0))
        while len(variables) > 0:
            clause += " or {}x[{}]".format(
                random.choice(["", "not "]), variables.pop(0)
            )
        cnf += "({}) and ".format(clause)
    # quick-fix for pattern that does not affect the result
    cnf += str(1)
    return lambda x: eval(cnf), cnf


def main():
    # define objective function: SAT in CNF
    n_var = 30
    fn, desc = generate_cnf(n_var, 20)

    def obj_func(x):
        return fn([1 if i in x else 0 for i in range(n_var)])

    # instantiate the optimizer
    opt = Combination(obj_func, list(range(n_var)))

    # define a threshold of acceptance
    threshold = 1

    # Uncomment line below to display the objective function
    # print("Objective function: \n{}".format(desc)

    # maximize HALF-SAT
    # (half of the number of variables are one and the rest is zero)
    opt.maximize(selection_size=n_var // 2, verbose=1, threshold=threshold)


if __name__ == "__main__":
    main()
