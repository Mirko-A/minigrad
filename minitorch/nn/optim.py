from abc import ABC, abstractmethod
from minitorch.matrix import Matrix

class Optimizer:
    def __init__(self, parameters: Matrix, learning_rate: float) -> None:
        self.parameters = parameters.flatten().item()
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self) -> None:
        ...

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = 0.0

class SGD(Optimizer):
    def __init__(self, parameters: Matrix, learning_rate: float) -> None:
        super().__init__(parameters, learning_rate)

    def step(self) -> None:
        for p in self.parameters:
            p.data -= p.grad * self.learning_rate