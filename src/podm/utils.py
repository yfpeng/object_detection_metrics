from typing import Any, List


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def _default_argument(arg: Any) -> List:
    if arg is None:
        return []
    if _isArrayLike(arg):
        return arg
    else:
        return [arg]
