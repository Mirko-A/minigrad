from minitorch.nn.module import Linear
from minitorch.nn.module import Sequence
from minitorch.nn.optim import Adam
from minitorch.nn.optim import SGD
from minitorch.tensor import Tensor

import torch

import numpy as np

def main1():
    t = Tensor.arange(1, 10)
    print(t.sqrt())

def main():
    m0 = Linear(2, 4)
    m1 = Linear(4, 2)
    m2 = Linear(2, 2)

    params = []
    params += m0.params()
    params += m1.params()
    params += m2.params()

    adam = Adam(params, 0.05)

    input = Tensor([[0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1]])
    target = Tensor([[0], 
                           [1], 
                           [1],
                           [1]])

    for epoch in range(50):
        adam.zero_grad()

        x0 = m0(input)
        x1 = m1(x0)
        x2 = m2(x1)
        pred = x2.sigmoid()

        l = ((target - pred) ** 2).sum()
        l.backward()
        print(f"Loss: {l}")
        #print(pred)

        adam.step()

if __name__ == "__main__":
    main()