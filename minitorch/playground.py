from minitorch.nn.module import Linear
from minitorch.nn.module import Sequence
from minitorch.nn.optim import Adam
from minitorch.nn.module import Sigmoid, MSELoss, CrossEntropyLoss
from minitorch.tensor import Tensor
from random import randint

import time

import torch
import torch.nn as nn
import numpy as np

# 0 1 2
# 3 4 5
# 6 7 8

import random

def multinomial_np(probabilities, num_samples=1):
    # Check if probabilities sum to 1
    if not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")

    # Number of categories
    num_categories = len(probabilities)

    # Generate random samples
    samples = np.random.multinomial(num_samples, probabilities)

    return samples

def multinomial(probabilities, num_samples=1):
    # Check if probabilities sum to 1
    if not sum(probabilities) == 1.0:
        raise ValueError("Probabilities must sum to 1.")

    # Number of categories
    num_categories = len(probabilities)

    # Generate random samples
    samples = []

    for _ in range(num_samples):
        rand_num = random.random()
        cumulative_prob = 0.0

        for i in range(num_categories):
            cumulative_prob += probabilities[i]
            if rand_num < cumulative_prob:
                samples.append(i)
                break

    return samples

# Example usage:
probabilities = [0.15, 0.25, 0.05, 0.10, 0.20, 0.12, 0.08, 0.03, 0.02]
num_samples = 1000

samples = multinomial(probabilities, num_samples)
samples_np = multinomial_np(probabilities, num_samples)

# print(f'KLOT:\n{samples}\nNAMPAJ:\n{samples_np}')


def main1():
    start = time.time()

    t1 = Tensor([[0.0, 0.0, 0.0, 0.25, 0.5, 0.25, 0.25, 0.0, 0.0, 0.0],
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [0.0, 0.0, 0.0, 0.25, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]])
    # t1 = Tensor.arange(0, 2*10*10, True).reshape([2, 10, 10])
    # t1 = torch.arange(0, 2*10*10, dtype=torch.float32).reshape([2, 10, 10])
    # print(t1)
    print(t1.softmax())
    print(t1.softmax().multinomial(num_samples=4))

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
    main1()