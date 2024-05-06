from __future__ import annotations
from typing import Optional
import math

from minitorch.storage import Storage
from minitorch.tensor import Function
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
    def forward(self, x: Storage, axis: tuple[int, ...]):
        self.input_shape = x.shape

        return x.sum(axis)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.expand(self.input_shape)

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

        return x ** y
    
    def backward(self, chain_grad: Storage) -> tuple[Optional[Storage], Optional[Storage]]:
        if self.was_exp:
            return chain_grad * (self.exp * (self.base ** (self.exp - Storage.full_like(self.exp, 1.0)))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base ** self.exp) * self.base.log()) if self.inputs_need_grad[1] else None
        else:
            return chain_grad * (self.exp * (self.base ** (self.exp - Storage.full_like(self.exp, 1.0)))) if self.inputs_need_grad[0] else None, \
                    chain_grad * ((self.base ** self.exp) * self.base.log()) if self.inputs_need_grad[1] else None


#* Ternary operations

class Where(Function):
    def forward(self, condition: Storage, x: Storage, y: Storage) -> Storage:
        self.condition = condition
        
        return Storage.where(condition, x, y)
    
    def backward(self, chain_grad: Storage) -> tuple[None, Optional[Storage], Optional[Storage]]:
        return None, \
                Storage.where(self.condition, chain_grad, chain_grad.const(0, chain_grad.dtype)) if self.inputs_need_grad[1] else None, \
                Storage.where(self.condition, chain_grad.const(0, chain_grad.dtype), chain_grad) if self.inputs_need_grad[2] else None


#* Movement operations

class Permute(Function):
    def forward(self, x: Storage, order: tuple[int, ...]) -> Storage:
        self.input_order = helpers.argsort(order)

        return x.permute(order)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.permute(self.input_order)

class Flip(Function):
    def forward(self, x: Storage, axis: tuple[int, ...]) -> Storage:
        self.arg = tuple([-1 if i in set(axis) else 1 for i in range(len(x.shape))])

        return x.stride(self.arg)
    
    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.stride(self.arg)

#* Mutate operations

class Reshape(Function):
    def forward(self, x: Storage, new_shape: tuple[int, ...]) -> Storage:
        self.input_shape = x.shape
        return x.reshape(new_shape)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.reshape(self.input_shape)


class Pad(Function):
    def forward(self, x: Storage, pad_sizes: tuple[tuple[int, int], ...]) -> Storage:
        self.arg = tuple([(ps[0], s+ps[0]) for s,ps in zip(x.shape, pad_sizes)])

        return x.pad(pad_sizes)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.slice(self.arg)

class Slice(Function):
    def forward(self, x: Storage, slice_sizes: tuple[tuple[int, int], ...]) -> Storage:
        self.arg = tuple([(ss[0], s-ss[1]) for s,ss in zip(x.shape, slice_sizes)])

        return x.slice(slice_sizes)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.pad(self.arg)

class Expand(Function):
    def forward(self, x: Storage, new_shape: tuple[int, ...]) -> Storage:
        self.sum_axes = tuple(i for i, (s_old, s_new) in enumerate(zip(x.shape, new_shape)) if s_old != s_new)
    
        return x.expand(new_shape)

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad.sum(self.sum_axes)
    
#* Activation functions

class Sigmoid(Function):
    def forward(self, x: Storage) -> Storage:
        self.result = Storage.full_like(x, 1) / (Storage.full_like(x, 1) + Storage.full_like(x, math.e) ** -x)

        return self.result

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad * (self.result * (Storage.full_like(self.result, 1) - self.result))

class Relu(Function):
    def forward(self, x: Storage) -> Storage:
        self.result = x.max(Storage.full_like(x, 0.0))
        return self.result

    def backward(self, chain_grad: Storage) -> Storage:
        return chain_grad * Storage.where(self.result > Storage.full_like(self.result, 0), self.result, Storage.full_like(self.result, 1))
