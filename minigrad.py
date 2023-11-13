from __future__ import annotations
from typing import Union

import math

class Value:
    def __init__(self, data: float) -> None:
        self._data = data
        self.grad = 0.0
        self.backward = lambda : None

    @property
    def data(self) -> float:
        return self._data

    # NOTE: Check if x, y can be omitted from functions below
    # NOTE: Should these functions be a class?

    # TODO: Add children (participants in operation) to each operation - Prki
    # TODO: Add backward functions to everything - Split up

    def _add(self, other: Union[Value, float], reverse=False) -> Value:
        x = self
        y = other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data + y.data)
        
        return out

    def _sub(self, other: Union[Value, float], reverse=False) -> Value:
        x = self
        y = other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data - y.data)
        
        return out
    
    def _neg(self) -> Value:
        out = Value(-self.data)
        return out
    
    def _mul(self, other: Union[Value, float], reverse=False) -> Value:
        x = self
        y = other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data * y.data)
        
        return out
    
    def _div(self, other: Union[Value, float], reverse=False) -> Value:
        x = self
        y = other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data / y.data)
        
        return out

    def _pow(self, other: Union[Value, float], reverse=False) -> Value:
        x = self
        y = other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data ** y.data)
        
        return out

    def __add__(self, other): return self._add(other)
    def __radd__(self, other): return self._add(other, True)
    
    def __sub__(self, other): return self._sub(other)
    def __rsub__(self, other): return self._sub(other, True)

    def __neg__(self): return self._neg()

    def __mul__(self, other): return self._mul(other)
    def __rmul__(self, other): return self._mul(other, True)
    
    def __truediv__(self, other): return self._div(other)
    def __rtruediv__(self, other): return self._div(other, True)
    
    def __pow__(self, other): return self._pow(other)
    def __rpow__(self, other): return self._pow(other, True)
    
    def exp(self):
        out = Value(math.e ** self.data)
        return out

    def sigmoid(self):
        out = 1 / (1 + (-self).exp())
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0)
        return out
    
    def __repr__(self) -> str:
        return f"Value: {self.data}"
    
a = Value(3.0)
b = Value(5.0)

c = a + b
d = c + 5.0
e = 10.0 + d
f = 15.0 + e
g = b ** a
h = b ** 2
i = a.relu()
print(i.data)
a = 3.0
