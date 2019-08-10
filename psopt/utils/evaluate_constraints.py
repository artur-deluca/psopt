import typing

Dict = typing.Dict[typing.Text, typing.Any]
List = typing.List[Dict]
Candidate = typing.List[int]


def evaluate_constraints(
    constraints: typing.Union[Dict, List], candidate: Candidate
) -> float:
    """Calculates potential penalties associated with the given candidate

    The optimization problem may be subjected to constraints.
    In order to make the algorithm find suitable solutions,
    a penalization function is employed to decrease the chances of
    unsuited solutions being the fittest.

    Args:
        constraints: A dict or list of dicts with the following keys:
            'fn': function or method to evaluate constraint
            'type': type of logical relationship implied between the fn output and 'value'
            'value': the expected value
        candidate: A list of ints representing the a possible solution candidate

    Returns:
        A float corresponding to the penalization factor
    """
    phi = 0

    operators = {
        "==": lambda x, y: (x - y) ** 2,
        "!=": lambda x, y: max(0, x - y) ** 2 + max(0, -x + y) ** 2,
        ">": lambda x, y: max(0, x - y) ** 2,
        ">=": lambda x, y: max(0, x - y) ** 2,
        "<": lambda x, y: max(0, -x + y) ** 2,
        "<=": lambda x, y: max(0, -x + y) ** 2,
    }

    for var in constraints:
        if isinstance(var, dict):
            phi += operators[var["type"]](var["fn"](candidate), var["value"])
    return phi
