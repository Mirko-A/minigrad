from minitorch.nn.linear import Linear
from minitorch.nn.optim import SGD
from minitorch.tensor import Tensor

import torch

import numpy as np

def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))

def main():
    #print(b)
    #print(a @ b)
    # a = a.sum(1)
    # print(a.shape)

    # Given matrices A and B
    test = Tensor.fill((3, 3), 1.0)
    test = test + 2
    print(test)
    #A = Tensor.arange(0, 2*3*3*3).reshape((2, 3, 3, 3))
    #at = torch.arange(0, 2*3*3*3).reshape((2, 3, 3, 3))
    #B = Tensor.arange(0, 27).reshape((3, 3, 3))
    #bt = torch.arange(0, 27).reshape((3, 3, 3))

    #res = A.sum(1)
    #rest = at.sum(1)
    #print(res)
    #print(res.shape)
    #print(rest)
    #print(rest.shape)

if __name__ == "__main__":
    main()