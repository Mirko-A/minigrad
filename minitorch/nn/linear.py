import math

from minitorch.value import Value
from minitorch.tensor import Tensor
from minitorch.nn.module import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, need_bias: bool = True) -> None:
        k = 1/in_features
        sqrt_k = math.sqrt(k)

        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = need_bias
        self.weights = Tensor.uniform(in_features, out_features, -sqrt_k, sqrt_k)
        self.bias = Tensor.uniform(1, out_features, -sqrt_k, sqrt_k) if need_bias else None

    def forward(self, input: Tensor) -> Tensor:
        output = input.matmul(self.weights)

        if self.bias is not None:
            output = output + self.bias

        return output
    
    def parameters(self) -> Tensor:
        parameters = self.weights.flatten().item()
        
        if self.bias is not None:
            bias = self.bias.item()

            if isinstance(bias, Value):
                parameters = parameters + [bias]
            else:
                parameters = parameters + bias

        return Tensor([parameters])
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)