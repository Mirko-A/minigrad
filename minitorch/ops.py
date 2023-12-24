from __future__ import annotations
from typing import Optional
import math

from minitorch.tensor import Function
from minitorch.buffer import MiniBuffer
from minitorch import helpers

#* Unary operations

class Neg(Function):
    def forward(self, x: MiniBuffer) -> MiniBuffer: 
        return -x
    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer: 
        return -chain_grad

class Log(Function):
    def forward(self, x: MiniBuffer) -> MiniBuffer:
        self.x = x
        self.base = math.e

        return x.log()
    
    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad / (self.x * MiniBuffer.full_like(self.x, self.base).log())

class Log2(Function):
    def forward(self, x: MiniBuffer) -> MiniBuffer:
        self.x = x
        self.base = 2

        return x.log2()
    
    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad / (self.x * MiniBuffer.full_like(self.x, self.base).log())

#* Reduce operations

class Sum(Function):
    def forward(self, x: MiniBuffer, sum_axis: Optional[int] = None):
        self.input_shape = x.shape

        return x.sum(sum_axis)

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.expand(self.input_shape)

#* Binary operations

class Add(Function):
    def forward(self, x: MiniBuffer, y: MiniBuffer) -> MiniBuffer:
        self.x, self.y = x, y
        
        return x + y
    
    def backward(self, chain_grad: MiniBuffer) -> tuple[Optional[MiniBuffer], Optional[MiniBuffer]]:
        return chain_grad if self.inputs_need_grad[0] else None, \
                chain_grad if self.inputs_need_grad[1] else None
    
class Sub(Function):
    def forward(self, x: MiniBuffer, y: MiniBuffer) -> MiniBuffer:
        self.x, self.y = x, y
        
        return x - y
    
    def backward(self, chain_grad: MiniBuffer) -> tuple[Optional[MiniBuffer], Optional[MiniBuffer]]:
        return chain_grad if self.inputs_need_grad[0] else None, \
                -chain_grad if self.inputs_need_grad[1] else None

class Mul(Function):
    def forward(self, x: MiniBuffer, y: MiniBuffer) -> MiniBuffer:
        self.x, self.y = x, y
        
        return x * y
    
    def backward(self, chain_grad: MiniBuffer) -> tuple[Optional[MiniBuffer], Optional[MiniBuffer]]:
        return chain_grad * self.y if self.inputs_need_grad[0] else None, \
                chain_grad * self.x if self.inputs_need_grad[1] else None

class Div(Function):
    def forward(self, x: MiniBuffer, y: MiniBuffer) -> MiniBuffer:
        self.x, self.y = x, y
        
        return x / y
    
    def backward(self, chain_grad: MiniBuffer) -> tuple[Optional[MiniBuffer], Optional[MiniBuffer]]:
        return chain_grad / self.y if self.inputs_need_grad[0] else None, \
                -chain_grad * (self.x / (self.y * self.y)) if self.inputs_need_grad[1] else None

class Pow(Function):
    def forward(self, x: MiniBuffer, y: MiniBuffer, is_exp: bool) -> MiniBuffer:
        # Use cases:
        #   1) True  -> this is an exponentiation operation (N^x)
        #   1) False -> this is a power operation (x^N) 
        self.was_exp = is_exp
        if is_exp:
            self.base = x
            self.exp = y
        else:
            self.base = x
            self.exp = y

        return x ** y
    
    def backward(self, chain_grad: MiniBuffer) -> tuple[Optional[MiniBuffer], Optional[MiniBuffer]]:
        if self.was_exp:
            return chain_grad * (self.exp * (self.base ** (self.exp - MiniBuffer.full_like(self.exp, 1.0)))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base ** self.exp) * self.base.log(math.e)) if self.inputs_need_grad[1] else None
        else:
            return chain_grad * ((self.base ** self.exp) * self.base.log(math.e)) if self.inputs_need_grad[0] else None, \
                    chain_grad * (self.exp * (self.base ** (self.exp - MiniBuffer.full_like(self.exp, 1.0)))) if self.inputs_need_grad[1] else None

#* Movement operations

class Permute(Function):
    def forward(self, x: MiniBuffer, order: tuple[int, ...]) -> MiniBuffer:
        self.input_order = order

        return x.permute(order)

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.permute(helpers.argsort(self.input_order))

class Reshape(Function):
    def forward(self, x: MiniBuffer, new_shape: tuple[int, ...]) -> MiniBuffer:
        self.input_shape = x.shape

        return x.reshape(new_shape)

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.reshape(self.input_shape)

#* Reshape operations

#? NOTE: Mirko, 24. 12. 2023 
# These are different from the Reshape movement operation. These operations
# add/remove elements of the tensor whereas the Reshape operation just
# changes the shape without modifying the elements.

class Pad(Function):
    def forward(self, x: MiniBuffer, new_shape: tuple[int, ...]) -> MiniBuffer:
        self.input_shape = x.shape

        return x.pad(new_shape)

    # TODO: Mirko, 24. 12. 2023 
    # It makes sense that shrink is opposite of pad but this
    # has not been checked!
    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.shrink(self.input_shape)

class Shrink(Function):
    def forward(self, x: MiniBuffer, new_shape: tuple[int, ...]) -> MiniBuffer:
        self.input_shape = x.shape

        return x.shrink(new_shape)

    # TODO: Mirko, 24. 12. 2023  
    # It makes sense that pad is opposite of shrink but this
    # has not been checked!
    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.pad(self.input_shape)

class Expand(Function):
    def forward(self, x: MiniBuffer, expansion_axis: int, expanded_size: int) -> MiniBuffer:
        self.reduce_dim = expansion_axis

        return x.expand(expansion_axis, expanded_size)

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.sum(self.reduce_dim)

#* Activation functions

class Sigmoid(Function):
    def forward(self, x: MiniBuffer) -> MiniBuffer:
        self.result = MiniBuffer.full_like(x, 1) / (MiniBuffer.full_like(x, 1) + MiniBuffer.full_like(x, math.e) ** (-x))

        return self.result

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad * (self.result * (MiniBuffer.full_like(self.result, 1.0) - self.result))

class Relu(Function):
    def forward(self, x: MiniBuffer) -> MiniBuffer:
        self.result = x.max(MiniBuffer.full_like(x, 0.0))
        return self.result

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad * MiniBuffer.masked_fill(MiniBuffer.full_like(self.result, 0.0),
                                                   MiniBuffer.full_like(self.result, 0.0) < self.result,
                                                   1.0)

# class Sigmoid(Function):
#     def forward(self, input: Matrix) -> Matrix:
#         out_data = []
        
#         for row in input.data:
#             out_row = []

#             for value in row:
#                 out_row.append(value.sigmoid())

#             out_data.append(out_row)

#         return Matrix(out_data, input.requires_grad)
    
#     def __call__(self, input: Matrix):
#         return self.forward(input)

# class Relu(Function):
#     def forward(self, input: Matrix) -> Matrix:
#         out_data = []
        
#         for row in input.data:
#             out_row = []

#             for value in row:
#                 out_row.append(value.relu())

#             out_data.append(out_row)

#         return Matrix(out_data, input.requires_grad)
    
#     def __call__(self, input: Matrix):
#         return self.forward(input)
    
# class Tanh(Function):
#     def forward(self, input: Matrix) -> Matrix:
#         out_data = []
        
#         for row in input.data:
#             out_row = []

#             for value in row:
#                 out_row.append(value.tanh())

#             out_data.append(out_row)

#         return Matrix(out_data, input.requires_grad)

#     def __call__(self, input: Matrix):
#         return self.forward(input)
    
# class Softmax(Function):
#     def forward(self, input: Matrix, dim = 0) -> Matrix:
#         assert dim in Matrix._VALID_DIM_VALUES, "Invalid dimension value provided. Expected: 0 or 1."

#         input = input if dim == 1 else input.T()
#         input_exp = input.exp()
#         input_exp_sums = input_exp.sum(dim=1).item()
#         out_data = []
        
#         for row_exp, row_exp_sum in zip(input_exp.data, input_exp_sums):
#             out_row = []

#             for value_exp in row_exp:
#                 probability = value_exp / row_exp_sum
#                 out_row.append(probability)

#             out_data.append(out_row)

#         out_mat = Matrix(out_data, input.requires_grad)

#         return out_mat if dim == 1 else out_mat.T()

#     def __call__(self, input: Matrix):
#         return self.forward(input)
    
# # Cost functions
    
# class CrossEntropyLoss(Function):
#     def forward(self, input: Matrix, target: Matrix) -> Matrix:
#         assert isinstance(target, Matrix), f"Cannot perform Cross-Entropy on target type {type(target)}"
#         assert input._dims_match_with(target), "Cannot perform Cross-Entropy. Dimensions of input don't match with target."
#         # NOTE: PyTorch uses base e here, might be relevant later
#         input_log = input.log(2)
#         out_data = []
        
#         for target_row, input_log_row in zip(target.data, input_log.data):
#             cross_entropy_sum = 0
            
#             for target_value, input_log_value in zip(target_row, input_log_row):
#                 cross_entropy = target_value * input_log_value
#                 cross_entropy_sum += cross_entropy
                
#             out_data.append(-cross_entropy_sum)
            
#         return Matrix([out_data], input.requires_grad)
    
#     def __call__(self, input: Matrix, target: Matrix):
#         return self.forward(input, target)
    
# class MSELoss(Function):
#     def forward(self, input: Matrix, target: Matrix) -> Matrix:
#         assert isinstance(target, Matrix), f"Cannot perform MSE on target type {type(target)}"
#         assert input._dims_match_with(target), "Cannot perform MSE. Dimensions of input don't match with target."
        
#         MSE = []
        
#         for input_row, target_row in zip(input.data, target.data):
#             row_error_sum = 0
            
#             for input_value, target_value in zip(input_row, target_row):
#                 squared_error = (target_value - input_value) ** 2
#                 row_error_sum += squared_error
                
#             MSE.append(row_error_sum / input.shape.col)
        
#         return Matrix([MSE], input.requires_grad)
    
#     def __call__(self, input: Matrix, target: Matrix):
#         return self.forward(input, target)
