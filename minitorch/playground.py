from minitorch.nn.module import Linear
from minitorch.nn.module import Sequence
from minitorch.nn.optim import Adam
from minitorch.nn.module import Sigmoid, Relu, Tanh, MSELoss, CrossEntropyLoss
from minitorch.tensor import Tensor

import torch

import numpy as np

# 0 1 2
# 3 4 5
# 6 7 8

# 1 1 1

def main1():
    t = Tensor.arange(0, 9).reshape((3, 3))
    y = Tensor([0, 1, 0])
    print(t.softmax())
    print(t.softmax().cross_entropy(y, -1))

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

    for epoch in range(50):
        adam.zero_grad()

        pred = prki_net(input)

        l = loss(pred, target)
        l.backward()
        print(f"Loss: {l}")
        #print(pred)

        adam.step()

if __name__ == "__main__":
    main()