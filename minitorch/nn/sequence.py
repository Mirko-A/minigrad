from minitorch.nn.module import Module
from minitorch.ops import Function
from minitorch.matrix import Matrix

class Sequence(Module):
    def __init__(self, *components: Module | Function) -> None:
        self.components = components
        self._parameters = self.get_components_params()
        
    def forward(self, input: Matrix) -> Matrix:
        next_input = input
        
        for component in self.components:
            output = component(next_input)
            next_input = output
        
        return next_input
    
    def parameters(self) -> Matrix:
        return self._parameters
        
    def get_components_params(self) -> Matrix:
        params = []
        
        for component in self.components:
            if hasattr(component, 'parameters'):
                params.append(component.parameters())
                
        params_concat = Matrix.cat(params, dim=1)
        return params_concat
    
    def get_component(self, id):
        return self.components[id]
        
    def __call__(self, input: Matrix) -> Matrix:
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