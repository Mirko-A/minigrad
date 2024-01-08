from minitorch.nn.module import Linear
from minitorch.nn.module import Sequence
from minitorch.nn.optim import Adam
from minitorch.nn.module import Sigmoid, Relu, Tanh, MSELoss, CrossEntropyLoss, AttentionHead, MultiHeadAttention, Block
from minitorch.tensor import Tensor

import time

import torch
import torch.nn as nn
import numpy as np

# 0 1 2
# 3 4 5
# 6 7 8

# 1 1 1

def main2():
    # ah = AttentionHead(16, 8, 16)
    # t1 = Tensor.arange(0, 4*16*16).reshape([4, 16, 16])
    # t1 = t1/(4*16*16)
    # h = ah(t1)
    # print(h)
    mha = Block(16, 4, 16)
    t1 = Tensor.arange(0, 4*16*16).reshape([4, 16, 16])
    t1 = t1/(4*16*16)
    h = mha(t1)
    print(h)


def main1():
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

def main0():
    t = Tensor.arange(0, 3 * 3, True).reshape((3, 3))
    print(t)
    x = t.sum(0)
    print(x)

def main():
    prki_net = Sequence(
        Linear(128, 128),
        Linear(128, 64),
        Linear(64, 1),
        Sigmoid()
    )
    loss = MSELoss()

    adam = Adam(prki_net.params(), 0.05)

    input = Tensor.randn([4, 128])
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
    main2()