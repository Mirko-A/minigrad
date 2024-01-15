from __future__ import annotations
from typing import Optional
import math

from minitorch.tensor import Function
from minitorch.storage import Storage
from minitorch import helpers

#* Unary operations

class Neg(Function):
    def forward(self, x: Storage) -> Storage: 
        return -x
    def backward(self, chain_grad: Storage) -> Storage: 
        return -chain_grad

class Log(Function):
    def forward(self, x: Storage) -> Storage:
        self.x = x
        self.base = math.e

        return x.log()
    
    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad / (self.x * Storage.full_like(self.x, self.base).log())

class Log2(Function):
    def forward(self, x: Storage) -> Storage:
        self.x = x
        self.base = 2

        return x.log2()
    
    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad / (self.x * Storage.full_like(self.x, self.base).log())

#* Reduce operations

class Sum(Function):
    def forward(self, x: Storage, axis: int):
        self.sum_axis = axis
        self.sum_axis_size = x.shape[axis]

        return x.sum(axis)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.expand(self.sum_axis, self.sum_axis_size)

#* Binary operations

class Add(Function):
    def forward(self, x: Storage, y: Storage) -> Storage:
        return x + y
    
    def backward(self, chain_grad: Storage) -> tuple[Optional[Storage], Optional[Storage]]:
        return chain_grad if self.inputs_need_grad[0] else None, \
                chain_grad if self.inputs_need_grad[1] else None
    
class Sub(Function):
    def forward(self, x: Storage, y: Storage) -> Storage:
        return x - y
    
    def backward(self, chain_grad: Storage) -> tuple[Optional[Storage], Optional[Storage]]:
        return chain_grad if self.inputs_need_grad[0] else None, \
                -chain_grad if self.inputs_need_grad[1] else None

class Mul(Function):
    def forward(self, x: Storage, y: Storage) -> Storage:
        self.x, self.y = x, y
        
        return x * y
    
    def backward(self, chain_grad: Storage) -> tuple[Optional[Storage], Optional[Storage]]:
        return chain_grad * self.y if self.inputs_need_grad[0] else None, \
                chain_grad * self.x if self.inputs_need_grad[1] else None

class Div(Function):
    def forward(self, x: Storage, y: Storage) -> Storage:
        self.x, self.y = x, y
        
        return x / y
    
    def backward(self, chain_grad: Storage) -> tuple[Optional[Storage], Optional[Storage]]:
        return chain_grad / self.y if self.inputs_need_grad[0] else None, \
                -chain_grad * (self.x / (self.y * self.y)) if self.inputs_need_grad[1] else None

class Pow(Function):
    def forward(self, x: Storage, y: Storage, is_exp: bool) -> Storage:
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

        return x.pow(y)
    
    def backward(self, chain_grad: Storage) -> tuple[Optional[Storage], Optional[Storage]]:
        if self.was_exp:
            return chain_grad * (self.exp * (self.base.pow((self.exp - Storage.full_like(self.exp, 1.0))))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base.pow(self.exp)) * self.base.log()) if self.inputs_need_grad[1] else None
        else:
            return chain_grad * (self.exp * (self.base.pow((self.exp - Storage.full_like(self.exp, 1.0))))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base.pow(self.exp)) * self.base.log()) if self.inputs_need_grad[1] else None
                    

#* Movement operations

class Permute(Function):
    def forward(self, x: Storage, order: tuple[int, ...]) -> Storage:
        self.input_order = order

        return x.permute(order)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.permute(helpers.argsort(self.input_order))

class Reshape(Function):
    def forward(self, x: Storage, new_shape: list[int]) -> Storage:
        self.input_shape = x.shape

        return x.reshape(new_shape)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.reshape(self.input_shape)

#* Mutate operations

class Pad(Function):
    def forward(self, x: Storage, axis: int, pad_sizes: tuple[int, int]) -> Storage:
        self.pad_axis = axis
        self.pad_sizes = pad_sizes

        return x.pad(axis, pad_sizes)

    # TODO: Mirko, 24. 12. 2023 
    # It makes sense that shrink is opposite of pad but this
    # has not been checked!
    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.shrink(self.pad_axis, self.pad_sizes)

class Shrink(Function):
    def forward(self, x: Storage, axis: int, shrink_sizes: [int, int]) -> Storage:
        self.shrink_axis = axis
        self.shrink_sizes = shrink_sizes

        return x.shrink(axis, shrink_sizes)

    # TODO: Mirko, 24. 12. 2023  
    # It makes sense that pad is opposite of shrink but this
    # has not been checked!
    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.pad(self.shrink_axis, self.shrink_sizes)

class Expand(Function):
    def forward(self, x: Storage, axis: int, expanded_size: int) -> Storage:
        self.reduce_axis = axis

        return x.expand(axis, expanded_size)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.sum(self.reduce_axis)
    
#* Activation functions

class Sigmoid(Function):
    def forward(self, x: Storage) -> Storage:
        self.result = Storage.full_like(x, 1) / (Storage.full_like(x, 1) + Storage.full_like(x, math.e).pow(-x))

        return self.result

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad * (self.result * (Storage.full_like(self.result, 1.0) - self.result))

class Relu(Function):
    def forward(self, x: Storage) -> Storage:
        self.result = x.max(Storage.full_like(x, 0.0))
        return self.result

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad * Storage.masked_fill(Storage.full_like(self.result, 0.0),
                                                       self.result > 0,
                                                       1.0)

