import numpy as np
import inspect
import sys


def hamming(source, target):
    return np.count_nonzero(source != target)


def l2(source, target):
    return np.linalg.norm(np.array(source) - np.array(target))


reference = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
