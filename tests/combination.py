import random


class HSAT:
    def __init__(self, seed):
        self.candidates = list(range(20))
        self.clauses = 50
        self.threshold = 0
        self.selection_size = int(len(self.candidates) / 2)
        self.obj_func = self.generate_obj_func(seed=seed)

    def generate_obj_func(self, seed):
        fn = self.generate_cnf(len(self.candidates), self.clauses, seed=seed)
        return lambda x: fn([1 if i in x else 0 for i in range(len(self.candidates))])

    @staticmethod
    def generate_cnf(n_vars, clauses, k=3, seed=None):
        random.seed(seed)
        cnf = str()
        for _ in range(clauses):
            clause = str()
            variables = random.sample(list(range(n_vars)), n_vars)[:k]
            clause += random.choice(["", "not "]) + "x[{}]".format(variables.pop(0))
            while len(variables) > 0:
                clause += " or {}x[{}]".format(
                    random.choice(["", "not "]), variables.pop(0)
                )
            cnf += "({}) and ".format(clause)
        # quick-fix for pattern that does not affect the result
        cnf += str(1)
        return lambda x: eval(cnf)


class coSum:
    def __init__(self, seed):
        size = 20
        random.seed(seed)
        self.candidates = random.sample(list(range(1, size)), size - 1)
        self.selection_size = 5
        self.threshold = sum(sorted(self.candidates)[: self.selection_size])
        self.constraint = {"fn": self.mod, "type": "==", "value": 1}

    @staticmethod
    def obj_func(x):
        return sum(x)

    @staticmethod
    def mod(x):
        return sum(x) % 2
