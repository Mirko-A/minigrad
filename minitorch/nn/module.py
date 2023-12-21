from abc import ABC, abstractmethod
from minitorch.tensor import Tensor

class Module(ABC):
    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        ...

    @abstractmethod
    def parameters(self) -> Tensor:
        ...