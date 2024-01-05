from __future__ import annotations
from typing import Optional
import math

import Backend as cpp
from minitorch.tensor import Function
from minitorch import helpers

#* Unary operations

class Neg(Function):
    def forward(self, x: cpp.MiniBuffer) -> cpp.MiniBuffer: 
        return -x
    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer: 
        return -chain_grad

class Log(Function):
    def forward(self, x: cpp.MiniBuffer) -> cpp.MiniBuffer:
        self.x = x
        self.base = math.e

        return x.log()
    
    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad / (self.x * cpp.MiniBuffer.full_like(self.x, self.base).log())

class Log2(Function):
    def forward(self, x: cpp.MiniBuffer) -> cpp.MiniBuffer:
        self.x = x
        self.base = 2

        return x.log2()
    
    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad / (self.x * cpp.MiniBuffer.full_like(self.x, self.base).log())

#* Reduce operations

class Sum(Function):
    def forward(self, x: cpp.MiniBuffer, axis: int, keepdims: bool):
        self.sum_axis = axis
        self.sum_axis_size = x.get_shape()[axis]

        return x.sum(axis)

    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad.expand(self.sum_axis, self.sum_axis_size)

#* Binary operations

class Add(Function):
    def forward(self, x: cpp.MiniBuffer, y: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return x + y
    
    def backward(self, chain_grad: cpp.MiniBuffer) -> tuple[Optional[cpp.MiniBuffer], Optional[cpp.MiniBuffer]]:
        return chain_grad if self.inputs_need_grad[0] else None, \
                chain_grad if self.inputs_need_grad[1] else None
    
class Sub(Function):
    def forward(self, x: cpp.MiniBuffer, y: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return x - y
    
    def backward(self, chain_grad: cpp.MiniBuffer) -> tuple[Optional[cpp.MiniBuffer], Optional[cpp.MiniBuffer]]:
        return chain_grad if self.inputs_need_grad[0] else None, \
                -chain_grad if self.inputs_need_grad[1] else None

class Mul(Function):
    def forward(self, x: cpp.MiniBuffer, y: cpp.MiniBuffer) -> cpp.MiniBuffer:
        self.x, self.y = x, y
        
        return x * y
    
    def backward(self, chain_grad: cpp.MiniBuffer) -> tuple[Optional[cpp.MiniBuffer], Optional[cpp.MiniBuffer]]:
        return chain_grad * self.y if self.inputs_need_grad[0] else None, \
                chain_grad * self.x if self.inputs_need_grad[1] else None

class Div(Function):
    def forward(self, x: cpp.MiniBuffer, y: cpp.MiniBuffer) -> cpp.MiniBuffer:
        self.x, self.y = x, y
        
        return x / y
    
    def backward(self, chain_grad: cpp.MiniBuffer) -> tuple[Optional[cpp.MiniBuffer], Optional[cpp.MiniBuffer]]:
        return chain_grad / self.y if self.inputs_need_grad[0] else None, \
                -chain_grad * (self.x / (self.y * self.y)) if self.inputs_need_grad[1] else None

class Pow(Function):
    def forward(self, x: cpp.MiniBuffer, y: cpp.MiniBuffer, is_exp: bool) -> cpp.MiniBuffer:
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
    
    def backward(self, chain_grad: cpp.MiniBuffer) -> tuple[Optional[cpp.MiniBuffer], Optional[cpp.MiniBuffer]]:
        if self.was_exp:
            return chain_grad * (self.exp * (self.base.pow((self.exp - cpp.MiniBuffer.full_like(self.exp, 1.0))))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base.pow(self.exp)) * self.base.log()) if self.inputs_need_grad[1] else None
        else:
            return chain_grad * (self.exp * (self.base.pow((self.exp - cpp.MiniBuffer.full_like(self.exp, 1.0))))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base.pow(self.exp)) * self.base.log()) if self.inputs_need_grad[1] else None
                    

#* Movement operations

class Permute(Function):
    def forward(self, x: cpp.MiniBuffer, order: tuple[int, ...]) -> cpp.MiniBuffer:
        self.input_order = order

        return x.permute(order)

    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad.permute(helpers.argsort(self.input_order))

class Reshape(Function):
    def forward(self, x: cpp.MiniBuffer, new_shape: list[int]) -> cpp.MiniBuffer:
        self.input_shape = x.get_shape()

        return x.reshape(new_shape)

    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad.reshape(self.input_shape)

#* Mutate operations

class Pad(Function):
    def forward(self, x: cpp.MiniBuffer, axis: int, pad_sizes: tuple[int, int]) -> cpp.MiniBuffer:
        self.pad_axis = axis
        self.pad_sizes = pad_sizes

        return x.pad(axis, pad_sizes)

    # TODO: Mirko, 24. 12. 2023 
    # It makes sense that shrink is opposite of pad but this
    # has not been checked!
    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad.shrink(self.pad_axis, self.pad_sizes)

class Shrink(Function):
    def forward(self, x: cpp.MiniBuffer, axis: int, shrink_sizes: [int, int]) -> cpp.MiniBuffer:
        self.shrink_axis = axis
        self.shrink_sizes = shrink_sizes

        return x.shrink(axis, shrink_sizes)

    # TODO: Mirko, 24. 12. 2023  
    # It makes sense that pad is opposite of shrink but this
    # has not been checked!
    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad.pad(self.shrink_axis, self.shrink_sizes)

class Expand(Function):
    def forward(self, x: cpp.MiniBuffer, axis: int, expanded_size: int) -> cpp.MiniBuffer:
        self.reduce_axis = axis

        return x.expand(axis, expanded_size)

    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad.sum(self.reduce_axis)
    
#* Activation functions

class Sigmoid(Function):
    def forward(self, x: cpp.MiniBuffer) -> cpp.MiniBuffer:
        self.result = cpp.MiniBuffer.full_like(x, 1) / (cpp.MiniBuffer.full_like(x, 1) + cpp.MiniBuffer.full_like(x, math.e).pow(-x))

        return self.result

    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad * (self.result * (cpp.MiniBuffer.full_like(self.result, 1.0) - self.result))

class Relu(Function):
    def forward(self, x: cpp.MiniBuffer) -> cpp.MiniBuffer:
        self.result = x.max(cpp.MiniBuffer.full_like(x, 0.0))
        return self.result

    def backward(self, chain_grad: cpp.MiniBuffer) -> cpp.MiniBuffer:
        return chain_grad * cpp.MiniBuffer.masked_fill(cpp.MiniBuffer.full_like(self.result, 0.0),
                                                       self.result > 0,
                                                       1.0)

