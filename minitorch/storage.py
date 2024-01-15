from __future__ import annotations
from random import gauss, uniform
import numpy as np
import math

import Backend as cpp
from minitorch.dtype import Dtype
from minitorch.settings import DEBUG

class Storage:
    def __init__(self, data: float | int | list | cpp.MiniBufferF32 | cpp.MiniBufferI32, dtype: Dtype) -> None:
        if isinstance(data, (cpp.MiniBufferF32, cpp.MiniBufferI32)):
            self.buffer = data
        elif isinstance(data, float):
            self.buffer = cpp.MiniBufferF32([data], [1])
        elif isinstance(data, int):
            self.buffer = cpp.MiniBufferI32([data], [1])
        elif isinstance(data, list):
            self.buffer = Storage._np_load(data, dtype)
        else:
            assert False, \
                f"Cannot construct Storage with given data: {type(data)}. Expected: int | float | list(nested)."

        self.dtype = dtype

    @property
    def data(self) -> list[float] | list[int]:
        return self.buffer.get_data()
    
    @property
    def shape(self) -> list[int]:
        return self.buffer.get_shape()
    
    @property
    def rank(self) -> list[int]:
        return self.buffer.get_rank()

    #* Static Storage generation methods

    @staticmethod
    def arange(start: int, end: int, dtype: Dtype):
        if dtype == Dtype.Float:
            data = cpp.MiniBufferF32.arange(start, end)
        elif dtype == Dtype.Int:
            data = cpp.MiniBufferI32.arange(start, end)
        
        return Storage(data, dtype)
        
    @staticmethod
    def fill(shape: list[int], value: int | float, dtype: Dtype):
        if dtype == Dtype.Float:
            data = cpp.MiniBufferF32.fill(shape, float(value))
        elif dtype == Dtype.Int:
            data = cpp.MiniBufferI32.fill(shape, int(value))

        return Storage(data, dtype)

    @staticmethod
    def randn(shape: list[int], mean: float = 0.0, std_dev: float = 1.0) -> Storage:
        data = cpp.MiniBufferF32([gauss(mean, std_dev) for _ in range(math.prod(shape))], shape)
        return Storage(data, Dtype.Float)

    @staticmethod
    def full_like(input: Storage, value: int | float) -> Storage:
        if input.dtype == Dtype.Float:
            data = cpp.MiniBufferF32.full_like(input.buffer, int(value))
        elif input.dtype == Dtype.Int:
            data = cpp.MiniBufferI32.full_like(input.buffer, float(value))

        return Storage(data, input.dtype)
        
    @staticmethod
    def uniform(shape: list[int], low: float, high: float) -> Storage:
        data = cpp.MiniBufferF32([uniform(low, high) for _ in range(math.prod(shape))], shape)
        return Storage(data, Dtype.Float)
    
    @staticmethod
    def masked_fill(input: Storage, mask: list[bool], value: float | int) -> Storage:
        if input.dtype == Dtype.Float:
            data = cpp.MiniBufferF32.masked_fill(input.buffer, mask, value)
        elif input.dtype == Dtype.Int:
            data = cpp.MiniBufferI32.masked_fill(input.buffer, mask, value)

        return Storage(data, input.dtype)

    @staticmethod
    def replace(input: Storage, target: float | int, new: float | int) -> Storage:
        if input.dtype == Dtype.Float:
            data = cpp.MiniBufferF32.replace(input.buffer, target, new)
        elif input.dtype == Dtype.Int:
            data = cpp.MiniBufferI32.replace(input.buffer, target, new)

        return Storage(data, input.dtype)

    @staticmethod
    def tril(input: Storage, diagonal: int = 0) -> Storage:
        if input.dtype == Dtype.Float:
            data = cpp.MiniBufferF32.tril(input.buffer, diagonal)
        elif input.dtype == Dtype.Int:
            data = cpp.MiniBufferI32.tril(input.buffer, diagonal)

        return Storage(data, input.dtype)

    #* Movement methods

    def permute(self, order: tuple[int, ...]):
        return Storage(self.buffer.permute(order), self.dtype)

    def reshape(self, new_shape: list[int]) -> Storage:
        return Storage(self.buffer.reshape(new_shape), self.dtype)
    
    #* Mutate methods

    def pad(self, axis: int, pad_sizes: tuple[int, int]):
        return Storage(self.buffer.pad(axis, pad_sizes), self.dtype)

    def shrink(self, axis: int, shrink_sizes: tuple[int, int]):
        return Storage(self.buffer.shrink(axis, shrink_sizes), self.dtype)

    def expand(self, axis: int, expanded_size: int) -> Storage:
        return Storage(self.buffer.expand(axis, expanded_size), self.dtype)

    #* Unary operations

    def __neg__(self) -> Storage:
        return Storage(-self.buffer, self.dtype)
    
    def log(self) -> Storage:
        return Storage(self.buffer.log(), self.dtype)
    
    def log2(self) -> Storage:
        return Storage(self.buffer.log2(), self.dtype)

    #* Reduce operations
    
    def sum(self, axis) -> Storage:
        return Storage(self.buffer.sum(axis), self.dtype)

    #* Binary operations

    def __add__(self, other: Storage) -> Storage:
        return Storage(self.buffer + other.buffer, self.dtype)
    
    def __sub__(self, other: Storage) -> Storage:
        return Storage(self.buffer - other.buffer, self.dtype)
    
    def __mul__(self, other: Storage) -> Storage:
        return Storage(self.buffer * other.buffer, self.dtype)
    
    def __truediv__(self, other: Storage) -> Storage:
        return Storage(self.buffer / other.buffer, self.dtype)
    
    def pow(self, other: Storage) -> Storage:
        return Storage(self.buffer.pow(other.buffer), self.dtype)

    def max(self, other: Storage) -> Storage:
        return Storage(self.buffer.max(other.buffer), self.dtype)

    def __eq__(self, other) -> bool | list[bool]:
        if isinstance(other, Storage):
            return self.buffer == other.buffer
        elif isinstance(other, (int, float)):
            return self.buffer == other
        else:
            assert False, f"Invalid type for Storage equality: {type(other)}. Expected Storage, int or float."

    def __lt__(self, other) -> list[bool]:
        return self.buffer < other

    def __gt__(self, other) -> list[bool]:
        return self.buffer > other

    #* Utility

    @staticmethod
    def _np_load(data: list, dtype: Dtype) -> cpp.MiniBufferF32 | cpp.MiniBufferI32:
        def convert_dtype():
            if dtype == Dtype.Float:
                result = np.float32
            elif dtype == Dtype.Int:
                result = np.int32
            
            return result

        _np = np.array(data, dtype=convert_dtype())
        shape = []

        for dim in _np.shape:
            shape.append(dim)
        
        _np = _np.reshape(-1).tolist()
        if dtype == Dtype.Float:
            return cpp.MiniBufferF32(_np, shape)
        elif dtype == Dtype.Int:
            return cpp.MiniBufferI32(_np, shape)
    
    def is_scalar(self) -> bool:
        return self.buffer.is_scalar()
    
    def is_square(self) -> bool:
        return self.buffer.is_square()
    