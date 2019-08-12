"""
===================================================
Half-SAT problem
===================================================

Half-Sat Problem With Example

Half-SAT = {F | F is a CNF formula with 2n variables and there is a satisfying assignment
in which n variables are set to True and n variables are set to False}.

`Half-SAT is NP-Hard <https://compquiz.blogspot.com/2010/06/half-sat-problem-with-example.html>`_

"""

import random
from psopt import Combination


def generate_cnf(n_vars, clauses, k=3, seed=None):
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
