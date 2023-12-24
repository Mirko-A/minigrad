from abc import ABC, abstractmethod
import math

from minitorch.tensor import Tensor

class Module(ABC):
    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        ...

    @abstractmethod
    def params(self) -> Tensor:
        ...

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, need_bias: bool = True) -> None:
        k = 1/in_features
        sqrt_k = math.sqrt(k)

        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = need_bias
        self.weights = Tensor.uniform((in_features, out_features), -sqrt_k, sqrt_k)
        self.bias = Tensor.uniform((1, out_features), -sqrt_k, sqrt_k) if need_bias else None

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

class Sequence(Module):
    def __init__(self, *components: Module) -> None:
        self.components = components
        self._params = self.get_components_params()
        
    def forward(self, input: Tensor) -> Tensor:
        next_input = input
        
        for component in self.components:
            output = component(next_input)
            next_input = output
        
        return next_input
    
    def params(self) -> Tensor:
        return self._params
        
    def get_components_params(self) -> Tensor:
        params = []
        
        for component in self.components:
            # TODO: Mirko, 24. 12. 2023
            # This is wrong now, used to be used for function objects
            # (for example softmax) back when those were created using
            # the Function class as a base.
            if hasattr(component, 'params'):
                params += component.params()
                
        return params
    
    def get_component(self, id):
        return self.components[id]
        
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    
    def __repr__(self) -> str:
        repr = str("Sequence(\n")
        
        for i, component in enumerate(self.components):
            repr += f"    ({i}): {type(component).__name__}("
            if hasattr(component, 'in_features'):
                repr += f"in_features={component.in_features}, out_features={component.out_features}, need_bias={component.need_bias}"
            repr += ")\n"
            
        repr += ")"
        
        return repr