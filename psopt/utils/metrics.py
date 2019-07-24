import inspect
import sys
import typing

import numpy as np

# =============================================================
#                      Built-in metrics
# =============================================================


def hamming(source, target):
    return np.count_nonzero(source != target)


def l2(source, target):
    return np.linalg.norm(np.array(source) - np.array(target))


# =============================================================
#                       Helper structure
# =============================================================

reference = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
M = typing.Union[typing.Text, typing.Callable, typing.List]


def unpack_metrics(
    selected_metrics: M
) -> typing.Dict[typing.Text, typing.Callable]:

    metrics_dict = dict()

    if isinstance(selected_metrics, str):
        metrics_dict.update({selected_metrics: reference[selected_metrics]})

    elif inspect.isfunction(selected_metrics):
        metrics_dict.update({selected_metrics.__name__: selected_metrics})

    elif isinstance(selected_metrics, list):
        for item in selected_metrics:
            metrics_dict.update(unpack_metrics(item))

    return metrics_dict
