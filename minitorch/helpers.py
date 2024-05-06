from typing import Iterable

# https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py

# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x: tuple[int, ...]): 
    return type(x)(sorted(range(len(x)), key=x.__getitem__))

def argfix(*x): 
    return list(x[0]) if x and x[0].__class__ in (tuple, list) else x

def flatten(i: Iterable) -> list: 
    return [item for sublist in i for item in sublist]

def all_int(i: Iterable) -> bool: 
    return all(isinstance(s, int) for s in i)

def float_equal(x: float, y: float) -> bool:
    assert isinstance(x, float) and isinstance(y, float), \
        f"Cannot compare {type(x)} and {type(y)}. Expected: float and float."

    return abs(x - y) < 1e-6