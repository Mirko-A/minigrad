from __future__ import annotations
from abc import ABC, abstractmethod

from minitorch.matrix import Matrix

# Sometimes it is useful to have the activation/cost functions
# in the object form as opposed to their functional form.
# That is the purpose of this file.

class Function(ABC):
    @abstractmethod
    def forward(self) -> Matrix:
        ...

# Activation functions

class Sigmoid(Function):
    def forward(self, input: Matrix) -> Matrix:
        out_data = []
        
        for row in input.data:
            out_row = []

            for value in row:
                out_row.append(value.sigmoid())

            out_data.append(out_row)

        return Matrix(out_data, input.requires_grad)
    
    def __call__(self, input: Matrix):
        return self.forward(input)

class Relu(Function):
    def forward(self, input: Matrix) -> Matrix:
        out_data = []
        
        for row in input.data:
            out_row = []

            for value in row:
                out_row.append(value.relu())

            out_data.append(out_row)

        return Matrix(out_data, input.requires_grad)
    
    def __call__(self, input: Matrix):
        return self.forward(input)
    
class Tanh(Function):
    def forward(self, input: Matrix) -> Matrix:
        out_data = []
        
        for row in input.data:
            out_row = []

            for value in row:
                out_row.append(value.tanh())

            out_data.append(out_row)

        return Matrix(out_data, input.requires_grad)

    def __call__(self, input: Matrix):
        return self.forward(input)
    
class Softmax(Function):
    def forward(self, input: Matrix, dim = 0) -> Matrix:
        assert dim in Matrix._VALID_DIM_VALUES, "Invalid dimension value provided. Expected: 0 or 1."

        input = input if dim == 1 else input.T()
        input_exp = input.exp()
        input_exp_sums = input_exp.sum(dim=1).item()
        out_data = []
        
        for row_exp, row_exp_sum in zip(input_exp.data, input_exp_sums):
            out_row = []

            for value_exp in row_exp:
                probability = value_exp / row_exp_sum
                out_row.append(probability)

            out_data.append(out_row)

        out_mat = Matrix(out_data, input.requires_grad)

        return out_mat if dim == 1 else out_mat.T()

    def __call__(self, input: Matrix):
        return self.forward(input)
    
# Cost functions
    
class CrossEntropyLoss(Function):
    def forward(self, input: Matrix, target: Matrix) -> Matrix:
        assert isinstance(target, Matrix), f"Cannot perform Cross-Entropy on target type {type(target)}"
        assert input._dims_match_with(target), "Cannot perform Cross-Entropy. Dimensions of input don't match with target."
        # NOTE: PyTorch uses base e here, might be relevant later
        input_log = input.log(2)
        out_data = []
        
        for target_row, input_log_row in zip(target.data, input_log.data):
            cross_entropy_sum = 0
            
            for target_value, input_log_value in zip(target_row, input_log_row):
                cross_entropy = target_value * input_log_value
                cross_entropy_sum += cross_entropy
                
            out_data.append(-cross_entropy_sum)
            
        return Matrix([out_data], input.requires_grad)
    
    def __call__(self, input: Matrix, target: Matrix):
        return self.forward(input, target)
    
class MSELoss(Function):
    def forward(self, input: Matrix, target: Matrix) -> Matrix:
        assert isinstance(target, Matrix), f"Cannot perform MSE on target type {type(target)}"
        assert input._dims_match_with(target), "Cannot perform MSE. Dimensions of input don't match with target."
        
        MSE = []
        
        for input_row, target_row in zip(input.data, target.data):
            row_error_sum = 0
            
            for input_value, target_value in zip(input_row, target_row):
                squared_error = (target_value - input_value) ** 2
                row_error_sum += squared_error
                
            MSE.append(row_error_sum / input.shape.col)
        
        return Matrix([MSE], input.requires_grad)
    
    def __call__(self, input: Matrix, target: Matrix):
        return self.forward(input, target)
