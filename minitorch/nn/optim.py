from abc import ABC, abstractmethod
from minitorch.tensor import Tensor
import math

class Optimizer:
    def __init__(self, parameters: Tensor, learning_rate: float) -> None:
        self.parameters = parameters.flatten().item()
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self) -> None:
        ...

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = 0.0

class SGD(Optimizer):
    def __init__(self, parameters: Tensor, learning_rate: float) -> None:
        super().__init__(parameters, learning_rate)

    def step(self) -> None:
        for p in self.parameters:
            p.data -= p.grad * self.learning_rate
                
class Adam(Optimizer):
    def __init__(self, parameters: Tensor, learning_rate: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08) -> None:
        super().__init__(parameters, learning_rate)
        self.betas = betas
        self.eps = eps
        self.prev_moments = [[0, 0] for _ in range(parameters.shape.col)]
        
    def step(self) -> None:
        for i, p in enumerate(self.parameters):
            beta1, beta2 = self.betas[0], self.betas[1]
            mt_prev, vt_prev = self.prev_moments[i][0], self.prev_moments[i][1]
            
            mt = beta1 * mt_prev + (1 - beta1) * p.grad
            vt = beta2 * vt_prev + (1 - beta2) * p.grad ** 2
            p.data -= (self.learning_rate / (math.sqrt(vt) + self.eps)) * mt
            
            self.prev_moments[i][0] = mt
            self.prev_moments[i][1] = vt