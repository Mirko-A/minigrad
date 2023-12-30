from minitorch.nn.module import Linear
from minitorch.nn.module import Sequence
from minitorch.nn.optim import Adam
from minitorch.nn.module import Sigmoid, Relu, Tanh, MSELoss, CrossEntropyLoss
from minitorch.tensor import Tensor
from minitorch.buffer import MiniBuffer

import time

import torch
import numpy as np

import cpp_backend

# 0 1 2
# 3 4 5
# 6 7 8

# 1 1 1

def main1():
    start = time.time()

    t = Tensor.arange(0, 8, True).reshape([1, 1, 8])
    #print(t)
    #t2 = Tensor.arange(0, 16).reshape([2, 2, 2, 2])

    #print(t)
    t2 = t.expand([2, 2, 8])
    print(t2)
    t3 = t2.sum()
    print(t3)
    t3.backward()

    print(f"Time: {time.time() - start}")

def main0():
    t = Tensor.arange(0, 3 * 3, True).reshape((3, 3))
    print(t)
    x = t.sum(0)
    print(x)

def main():
    prki_net = Sequence(
        Linear(2, 4),
        Linear(4, 2),
        Linear(2, 2),
        Sigmoid()
    )
    loss = MSELoss()

    adam = Adam(prki_net.params(), 0.05)

    input = Tensor([[0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1]])
    target = Tensor([[0], 
                     [1], 
                     [1],
                     [1]])

    start = time.time()

    for epoch in range(500):
        adam.zero_grad()

        pred = prki_net(input)

        l = loss(pred, target)
        l.backward()
        #print(f"Loss: {l}")

        adam.step()

    print(pred)
    print(l)

    print(f"Total time (new code): {time.time() - start}")

if __name__ == "__main__":
    main()