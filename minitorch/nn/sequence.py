from minitorch.nn.module import Module
from minitorch.matrix import Matrix

class Sequence(Module):
    def __init__(self, modules: list) -> None:
        self.modules = modules
        
    def forward(self, input: Matrix) -> Matrix:
        next_input = input
        
        for module in self.modules:
            ...
        
    def __call__(self, input: Matrix) -> Matrix:
        return self.forward(input)