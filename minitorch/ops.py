from abc import ABC, abstractmethod
from minitorch.matrix import Matrix

class Function:
    @abstractmethod
    def forward(self, input: Matrix) -> Matrix:
        ...

    @abstractmethod
    def parameters(self) -> Matrix:
        ...