from abc import ABC, abstractmethod
from typing import Optional
import math

from minitorch.tensor import Tensor

# Base class for all NN modules

class Module(ABC):
    @abstractmethod
    def forward(self, input) -> Tensor:
        ...

    @abstractmethod
    def params(self) -> Optional[list[Tensor]]:
        ...

# Simple linear module (y = ax + b)

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, need_bias: bool = True) -> None:
        k = 1/in_features
        sqrt_k = math.sqrt(k)

        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = need_bias
        self.weights = Tensor.uniform([in_features, out_features], -sqrt_k, sqrt_k)
        self.bias = Tensor.uniform([1, out_features], -sqrt_k, sqrt_k) if need_bias else None

    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weights

        if self.bias is not None:
            output = output + self.bias

        return output
    
    def params(self) -> list[Tensor]:
        params = [self.weights]
        
        if self.bias is not None:
            params.append(self.bias)

        return params
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

# Activation function modules

class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()
    
    def params(self) -> None:
        return None

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()
    
    def params(self) -> None:
        return None

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

class Relu(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()
    
    def params(self) -> None:
        return None

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, input: Tensor) -> Tensor:
        return input.softmax(self.axis)
    
    def params(self) -> None:
        return None

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

# Loss function modules

class MSELoss(Module):
    def __init__(self, axis: Optional[int] = None) -> None:
        self.axis = axis

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.MSE(target, self.axis)
    
    def params(self) -> None:
        return None

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)

class CrossEntropyLoss(Module):
    def __init__(self, axis: Optional[int] = None) -> None:
        self.axis = axis

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.cross_entropy(target, self.axis)
    
    def params(self) -> None:
        return None

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)

# Embedding module (lookup table)

class Embedding(Module):
    def __init__(self, n_embeddings: int, embedding_dim: int):
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = [Tensor.randn([1, embedding_dim]) for _ in range(n_embeddings)]

    def forward(self, input: int) -> Tensor:
        return self.embeddings[input]
    
    def params(self) -> list[Tensor]:
        return self.embeddings
    
    def __call__(self, input: int) -> Tensor:
        return self.forward(input)

# Sequence module. Used to create chains of basic modules

class Sequence(Module):
    def __init__(self, *modules: Module) -> None:
        self.modules = modules
        self._params = self._get_module_params()
        
    def forward(self, input: Tensor) -> Tensor:
        next_input = input
        
        for component in self.modules:
            output = component(next_input)
            next_input = output
        
        return next_input
    
    def params(self) -> list[Tensor]:
        return self._params
        
    def _get_module_params(self) -> list[Tensor]:
        params = []
        
        for module in self.modules:
            p = module.params()
            if p is not None:
                params += p 
                
        return params
    
    def get_module(self, id):
        return self.modules[id]
        
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    
    def __repr__(self) -> str:
        repr = str("Sequence(\n")
        
        for i, module in enumerate(self.modules):
            repr += f"    ({i}): {type(module).__name__}("
            if hasattr(module, 'in_features'):
                repr += f"in_features={module.in_features}, out_features={module.out_features}, need_bias={module.need_bias}"
            repr += ")\n"
            
        repr += ")"
        
        return repr
