from __future__ import annotations
from enum import Enum, auto
from typing import Optional

import numpy as np

class Storage:
    Value = bool | int | float 
    ArrayLike = list | np.ndarray
    
    class Dtype(Enum):
        Bool = 0,
        Byte = auto(),
        Short = auto(),
        Int = auto(),
        Long = auto(),
        Float = auto(),
        Double = auto(),
        
        def to_np_dtype(self):
            match self.name:
                case "Bool":
                    return np.bool_
                case "Byte":
                    return np.uint8
                case "Short":
                    return np.int16
                case "Int":
                    return np.int32
                case "Long":
                    return np.int64
                case "Float":
                    return np.float32
                case "Double":
                    return np.float64
                
        @staticmethod
        def from_np_dtype(dtype):
            match dtype:
                case "bool_":
                    return Storage.Dtype.Bool
                case "uint8":
                    return Storage.Dtype.Byte
                case "int16":
                    return Storage.Dtype.Short
                case "int32":
                    return Storage.Dtype.Int
                case "int64":
                    return Storage.Dtype.Long
                case "float32":
                    return Storage.Dtype.Float
                case "float64":
                    return Storage.Dtype.Double
                case _:
                    assert False, \
                        f"ERROR: Unsupported Dtype: {dtype}."
                
    def __init__(self, data: Value | ArrayLike, dtype: Optional[Dtype] = None) -> None:
        if dtype:
            self._np = np.array(data, dtype=dtype.to_np_dtype())
        else:
            self._np = np.array(data)
        self.dtype = Storage.Dtype.from_np_dtype(self._np.dtype)
    
    @property
    def shape(self):
        return self._np.shape
    
    #* Storage generation functions 
    
    @staticmethod
    def full(shape: tuple[int, ...], value: Value, dtype: Optional[Dtype] = None):
        return Storage(np.full(shape=shape, fill_value=value), dtype)

    @staticmethod
    def full_like(other: Storage, value: Value, dtype: Optional[Dtype] = None) -> Storage:
        return Storage(np.full_like(other._np, value), dtype)

    def const(self, value: Value, dtype: Optional[Dtype] = None) -> Storage:
        return Storage.full_like(self, value, dtype)

    @staticmethod
    def arange(start: int, end: int, dtype: Optional[Dtype] = None):
        return Storage(np.arange(start=start, stop=end), dtype)

    #* Unary functions

    def log(self) -> Storage:
        return Storage(np.log(self._np))
    
    def log2(self) -> Storage:
        return Storage(np.log2(self._np))
    
    def sum(self, axis: tuple[int, ...]) -> Storage:
        return Storage(np.sum(self._np, axis, keepdims=True))

    #* Binary functions
    
    def max(self, other: Storage) -> Storage:
        return Storage(np.maximum(self._np, other._np))

    def min(self, other: Storage) -> Storage:
        return Storage(np.minimum(self._np, other._np))

    #* Ternary functions
    
    @staticmethod
    def where(condition: Storage, x: Storage, y: Storage):
        return Storage(np.where(condition._np, x._np, y._np))

    #* Movement functions

    def permute(self, order: tuple[int, ...]) -> Storage:
        return Storage(np.transpose(self._np, axes=order))

    def stride(self, arg): 
        return Storage(self._np[tuple(slice(None, None, i) for i in arg)])


    #* Mutate operations
    
    def reshape(self, new_shape: tuple[int, ...]) -> Storage:
        return Storage(np.reshape(self._np, newshape=new_shape))

    def pad(self, pad_sizes: tuple[tuple[int, int], ...]) -> Storage:
        return Storage(np.pad(self._np, pad_width=pad_sizes))

    def slice(self, slice_sizes: tuple[tuple[int, int], ...]) -> Storage:
        return Storage(self._np[tuple(slice(ss[0], ss[1], None) for s, ss in zip(self.shape, slice_sizes))])
    
    def expand(self, new_shape: tuple[int, ...]) -> Storage:
        return Storage(np.broadcast_to(self._np, new_shape))
    
    # Special method overloads
    
    def __neg__(self) -> Storage:
        return Storage(-self._np)
    
    def __add__(self, other: Storage) -> Storage:
        return Storage(self._np + other._np)
    
    def __sub__(self, other: Storage) -> Storage:
        return Storage(self._np - other._np)
    
    def __mul__(self, other: Storage) -> Storage:
        return Storage(self._np * other._np)
    
    def __truediv__(self, other: Storage) -> Storage:
        return Storage(self._np / other._np)
    
    def __pow__(self, other: Storage) -> Storage:
        return Storage(self._np ** other._np)
    
    def __eq__ (self, other: Storage) -> bool:
        return Storage(self._np == other._np)
    def __ne__ (self, other: Storage) -> bool:
        return Storage(self._np != other._np)
    def __lt__ (self, other: Storage) -> bool:
        return Storage(self._np < other._np)
    def __gt__ (self, other: Storage) -> bool:
        return Storage(self._np > other._np)
    def __le__ (self, other: Storage) -> bool:
        return Storage(self._np <= other._np)
    def __ge__ (self, other: Storage) -> bool:
        return Storage(self._np >= other._np)
    
    # Utility
    
    def eq(self, other: Storage) -> bool:
        return np.array_equal(self._np, other._np)
    
    def is_scalar(self) -> bool:
        return self.shape == ()
    
    def is_square(self) -> bool:
        return self.shape[-1] == self.shape[-2]