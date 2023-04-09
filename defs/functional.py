import numpy as np

from typing import TypeVar, Callable, Iterable, Iterator, Union


T1 = TypeVar('T1')
T2 = TypeVar('T2')

B = Union[bool, int]
ComparisonFunction = Callable[[T1, T1], B]
ComparisonFunctionInt = Callable[[T1, T1], int]
ComparisonFunctionBool = Callable[[T1, T1], bool]


def cmp_to_key_conversion(cmp_fn: ComparisonFunction) -> ComparisonFunctionInt:
    def new_cmp_fn(a: T1, b: T1) -> int:
        res: B = cmp_fn(a, b)
        if isinstance(res, bool):
            return 1 if res is True else -1
        return res
    return new_cmp_fn


def negate_cmp(cmp_fn: ComparisonFunction) -> ComparisonFunction:
    def new_cmp_fn(a: T1, b: T1) -> B:
        res: B = cmp_fn(a, b)
        if isinstance(res, (bool, np.bool_)):
            return not res
        if isinstance(res, int):
            return -res
        raise TypeError(f'Cannot negate {cmp_fn} return of type {type(res)}')
    return new_cmp_fn


def paired_map(fn: Callable[[T1, T1], T2], seq: Iterable[T1]) -> T2:
    seq: Iterator[T1] = iter(seq)
    x: T1 = next(seq)
    for y in seq:
        yield fn(x, y)
        x = y
