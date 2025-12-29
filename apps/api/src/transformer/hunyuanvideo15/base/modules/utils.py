import collections
from itertools import repeat


def _ntuple(n):
    """Create a function that converts input to n-tuple."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


# Convenience functions for common tuple sizes
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
