from abc import ABC, abstractmethod

from minitorch.tensor import Tensor
from minitorch.dtype import Dtype
from minitorch import helpers

class Optimizer:
    def __init__(self, params: list[Tensor], learning_rate: float) -> None:
        for p in params:
            if not p.requires_grad:
                p.requires_grad = True

        self.params = params
        self.lr = Tensor([learning_rate])

    @abstractmethod
    def step(self) -> None:
        ...

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params: list[Tensor], 
                 learning_rate: float = 1e-3,
                 momentum: float = 0.0, 
                 weight_decay: float = 0.0,
                 nesterov: bool = False) -> None:
        super().__init__(params, learning_rate)
        self.momentum = momentum
        self.wd = weight_decay
        self.nesterov = nesterov
        self.b = [Tensor.zeros(t.shape, dtype=Dtype.Float) for t in self.params] if not helpers.float_equal(self.momentum, 0.0) else None

    def step(self) -> None:
        for i, t in enumerate(self.params):
            assert t.grad is not None
            g = t.grad + self.wd * t.detach()

            if not helpers.float_equal(self.momentum, 0.0):
                self.b[i].assign(self.momentum * self.b[i] + g)

                if self.nesterov:
                    g = (g + self.momentum * self.b[i])
                else:
                    g = self.b[i]

            t.assign(t.detach() - t.grad * self.lr)
                
class Adam(Optimizer):
    def __init__(self, params: Tensor, 
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 betas: tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-08) -> None:
        super().__init__(params, learning_rate)
        self.wd = weight_decay
        self.b1, self.b2 = betas[0], betas[1]
        self.eps = eps
        self.m = [Tensor.zeros(t.shape, dtype=Dtype.Float) for t in self.params]
        self.v = [Tensor.zeros(t.shape, dtype=Dtype.Float) for t in self.params]
        
    def step(self) -> None:
        for i, t in enumerate(self.params):
            assert t.grad is not None
            g = t.grad

            self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
            self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (g * g))

            m_hat = self.m[i] / (1.0 - self.b1**(i + 1))
            v_hat = self.v[i] / (1.0 - self.b2**(i + 1))

            up = m_hat / (v_hat.sqrt() + self.eps)

            t.assign(t.detach() - self.lr * up + self.wd * t.detach())