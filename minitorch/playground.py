from minitorch.nn.module import Linear
from minitorch.nn.module import Sequence
from minitorch.nn.optim import Adam
from minitorch.nn.module import Sigmoid, Relu, Tanh, MSELoss, CrossEntropyLoss
from minitorch.tensor import Tensor
from minitorch import buffer

import time

import torch
import numpy as np

# 0 1 2
# 3 4 5
# 6 7 8

# 1 1 1

def main1():
    t = Tensor.arange(0, 3 * 3, True).reshape((3, 3))
    t = t.pad(0, [1, 1])
    print(t)
    mask = t == 0.0
    print(mask)

    t = Tensor.masked_fill(t, mask, 1.3)
    print(t)

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
        print(f"Loss: {l}")

        adam.step()

    
    print(pred)

    print(f"Total time (new code): {time.time() - start}")

if __name__ == "__main__":
    main()