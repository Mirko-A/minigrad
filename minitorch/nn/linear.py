import math

from minitorch.value import Value
from minitorch.matrix import Matrix
from minitorch.nn.module import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, need_bias: bool = True) -> None:
        k = 1/in_features
        sqrt_k = math.sqrt(k)

        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = need_bias
        self.weights = Matrix.uniform(in_features, out_features, -sqrt_k, sqrt_k)
        self.bias = Matrix.uniform(1, out_features, -sqrt_k, sqrt_k) if need_bias else None

    def forward(self, input: Matrix) -> Matrix:
        output = input.matmul(self.weights)

        if self.bias is not None:
            output = output + self.bias

        return output
    
    def parameters(self) -> Matrix:
        parameters = self.weights.flatten().item()
        
        if self.bias is not None:
            bias = self.bias.item()

            if isinstance(bias, Value):
                parameters = parameters + [bias]
            else:
                parameters = parameters + bias

        return Matrix([parameters])
    
    def __call__(self, input: Matrix) -> Matrix:
        return self.forward(input)