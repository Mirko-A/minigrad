# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x: tuple[int, ...]): 
    return type(x)(sorted(range(len(x)), key=x.__getitem__))