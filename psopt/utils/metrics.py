"""
Metrics
=======

This module constains the set of all available metrics to track during optimization
"""

import inspect
import sys
import typing

import numpy as np

# =============================================================
#                      Built-in metrics
# =============================================================


def hamming(source, target):
    """Calculates the Hamming distance between the particles position (source) and a given target.
    If no specific target is provided through functools.partial, target variable will be assigned to the global optimum position

    Returns:
        metrics observation at each iteration under attribute ``history`` on ``psopt.utils.Results`` object"""
    return np.count_nonzero(source != target)


def l2(source, target):
    """Calculates the L2-Norm between the particles position (source) and a given target.
    If no specific target is provided through functools.partial, target variable will be assigned to the global optimum position

    Returns:
        metrics observation at each iteration under attribute ``history`` on ``psopt.utils.Results`` object"""
    return np.linalg.norm(np.array(source) - np.array(target))


# =============================================================
#                       Helper structure
# =============================================================

reference = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
M = typing.Union[typing.Text, typing.Callable, typing.List]


def _unpack_metrics(selected_metrics: M) -> typing.Dict[typing.Text, typing.Callable]:

    metrics_dict = dict()

    if isinstance(selected_metrics, str):
        metrics_dict.update({selected_metrics: reference[selected_metrics]})

    elif inspect.isfunction(selected_metrics):
        metrics_dict.update({selected_metrics.__name__: selected_metrics})

    elif isinstance(selected_metrics, list):
        for item in selected_metrics:
            metrics_dict.update(_unpack_metrics(item))

    return metrics_dict
