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
    def forward(self, x: MiniBuffer, axis: int):
        self.sum_axis = axis
        self.sum_axis_size = x.shape[axis]

        return x.sum(axis)

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.expand(self.sum_axis, self.sum_axis_size)

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
                    chain_grad * ((self.base ** self.exp) * self.base.log()) if self.inputs_need_grad[1] else None
        else:
            return chain_grad * (self.exp * (self.base ** (self.exp - MiniBuffer.full_like(self.exp, 1.0)))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base ** self.exp) * self.base.log()) if self.inputs_need_grad[1] else None
                    

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
    def forward(self, x: MiniBuffer, axis: int, expanded_size: int) -> MiniBuffer:
        self.reduce_dim = axis

        return x.expand(axis, expanded_size)

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        return chain_grad.sum(self.reduce_dim)

class Cat(Function):
    def forward(self, x: MiniBuffer, y: MiniBuffer, axis: int) -> MiniBuffer:
        ...

    def backward(self, chain_grad: MiniBuffer) -> MiniBuffer:
        ...

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
                                                   self.result > 0,
                                                   1.0)
