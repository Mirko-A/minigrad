from minitorch.nn.module import Linear
from minitorch.nn.module import Sequence
from minitorch.nn.optim import Adam
from minitorch.nn.module import Sigmoid, Relu, Tanh, MSELoss, CrossEntropyLoss
from minitorch.tensor import Tensor

import time

import torch
import numpy as np

# 0 1 2
# 3 4 5
# 6 7 8

# 1 1 1

def main():
    start = time.time()

    t1 = Tensor.arange(0, 2*64*64, True).reshape([2, 64, 64])
    t1 = t1 / 1000
    t2 = Tensor.arange(0, 2*64*64, True).reshape([2, 64, 64])
    t2 = t2 / 1000
    #print(t)
    #t2 = Tensor.arange(0, 16).reshape([2, 2, 2, 2])

    for _ in range(100):
        y = t1 @ t2

    print(y)
    print(f"Time: {time.time() - start}")

def main1():
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
        print(f"Loss: {l}")

        adam.step()

    
    print(pred)

    print(f"Total time (new code): {time.time() - start}")

if __name__ == "__main__":
    main()