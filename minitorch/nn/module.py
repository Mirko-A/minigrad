from abc import ABC, abstractmethod
from typing import Optional
import math
from dataclasses import dataclass

from minitorch.tensor import Tensor

# Base class for all NN modules
class Module(ABC):
    def eval(self):
        params = self.params()
        if params is not None:
            for p in params:
                p.assign(Tensor(p._data))
                
    def train(self):
        params = self.params()
        if params is not None:
            for p in params:
                p.assign(Tensor(p._data, requires_grad=True))
    
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

# Normalization modules

class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        self.eps = eps
        self.gamma = Tensor.ones([dim])
        self.beta = Tensor.zeros([dim]) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(-1, keepdims=True)
        var = input.var(-1, keepdims=True)

        out = ((input - mean) / (var + self.eps).sqrt()) * self.gamma
        if self.beta is not None:
            out = out + self.beta

        return out

    def params(self) -> list[Tensor]:
        return [self.gamma, self.beta]

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

# Embedding module (lookup table)

class Embedding(Module):
    def __init__(self, n_embeddings: int, embedding_dim: int):
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = [Tensor.randn((1, embedding_dim)) for _ in range(n_embeddings)]

    def forward(self, input: Tensor) -> Tensor:
        embeddings = [self.embeddings[i] for i in input.flatten()._np]

        return Tensor.stack(embeddings).reshape(input.shape + (self.embedding_dim, ))
    
    def params(self) -> list[Tensor]:
        return self.embeddings
    
    def __call__(self, input: list[int]) -> Tensor:
        return self.forward(input)
 
class PositionalEncoding(Module):
    def __init__(self, d_model: int, N: int = 10_000):
        self.d_model = d_model
        self.N = N

    def forward(self, input: int) -> Tensor:
        L = input
        encodings = []

        for k in range(L):
            for i in range(self.d_model // 2):
                encodings.append(math.sin(k / (self.N ** ((2 * i) / self.d_model))))
                encodings.append(math.cos(k / (self.N ** ((2 * i) / self.d_model))))
        
        return Tensor(encodings).reshape([L, self.d_model])
    
    def params(self) -> None:
        return None
    
    def __call__(self, input: int) -> Tensor:
        return self.forward(input)


# Self-attention modules
 
class AttentionHead(Module):
    def __init__(self, embedding_dim: int, head_size: int, context_len: int):
        self.query = Linear(embedding_dim, head_size, need_bias=False)
        self.key = Linear(embedding_dim, head_size, need_bias=False)
        self.value = Linear(embedding_dim, head_size, need_bias=False)
        self.tril = Tensor.ones([context_len, context_len]).tril()
        
    def forward(self, input: Tensor) -> Tensor:
        assert len(input.shape) == 3, "BxTxC Tensor expected."
        B,T,C = input.shape
        q = self.query(input)
        k = self.key(input)

        w = q @ k.transpose() * k.shape[-1]**-0.5
        w = Tensor.masked_fill(w, self.tril[:T, :T] == 0, float('-inf'))
        
        w = w.softmax()
        
        v = self.value(input)
        out = w @ v
        return out
        
    def params(self) -> list[Tensor]:
        return self.query.params() + self.key.params() + self.value.params()
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    
class MultiHeadAttention(Module):
    def __init__(self, embedding_dim: int, num_heads: int, head_size: int, context_len: int):
        self.heads = [AttentionHead(embedding_dim, head_size, context_len) for _ in range(num_heads)]
        self.proj = Linear(head_size * num_heads, embedding_dim)

    def forward(self, input):
        out = Tensor.concat([h(input) for h in self.heads], -1)
        out = self.proj(out)
        return out
    
    def params(self) -> list[Tensor]:
        p = self.proj.params()
        for h in self.heads:
            p += h.params()
        return p
    
    def __call__(self, input: Tensor) -> Tensor:
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
