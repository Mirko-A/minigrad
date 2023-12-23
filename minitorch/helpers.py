# https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py

# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x: tuple[int, ...]): 
    return type(x)(sorted(range(len(x)), key=x.__getitem__))

def argfix(*x): 
    return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x