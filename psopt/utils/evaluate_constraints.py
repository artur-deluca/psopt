
def evaluate_constraints(constraints, candidate):
    phi = 0

    operators = {

        "==": lambda x, y: (x - y)**2,
        "!=": lambda x, y: max(0, x - y)**2 + max(0, -x + y)**2,
        ">": lambda x, y: max(0, x - y)**2,
        ">=": lambda x, y: max(0, x - y)**2,
        "<": lambda x, y: max(0, -x + y)**2,
        "<=": lambda x, y: max(0, -x + y)**2,
    }

    for var in constraints:
        if isinstance(var, dict):
            phi += operators[var['type']](var['fn'](candidate), var['value'])
    return phi
