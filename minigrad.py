from __future__ import annotations

import math

class Value:
    def __init__(self, data: float, _children=()) -> None:
        self._data = data
        self._children = _children
        self.grad = 0.0
        self._backward = lambda : None

    @property
    def data(self) -> float:
        return self._data

    # NOTE: Should these functions be a class?

    # TODO: Add backward functions to everything - Split up
    # TODO: Test if backward works for reverse functions

    def _add(self, other: Value | float) -> Value:
        if not isinstance(other, Value): other = Value(other)

        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def _sub(self, other: Value | float, reverse=False) -> Value:
        x, y = self, other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data - y.data, (self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        
        return out
    
    def _neg(self) -> Value:
        out = Value(-self.data)
        return out
    
    def _mul(self, other: Value | float) -> Value:
        if not isinstance(other, Value): other = Value(other)

        out = Value(self.data * other.data, (self, other))
        
        def _backward():
            self.grad += other.grad * out.grad
            other.grad += self.grad * out.grad
        out._backward = _backward
        
        return out
    
    def _div(self, other: Value | float, reverse=False) -> Value:
        x, y = self, other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data / y.data, (self, other))
        
        return out

    def _pow(self, other: Value | float, reverse=False) -> Value:
        x, y = self, other if isinstance(other, Value) else Value(other)
        
        if reverse:
            x, y = y, x

        out = Value(x.data ** y.data, (self, other))
        
        return out

    def __add__(self, other): return self._add(other)
    def __radd__(self, other): return self._add(other)
    
    def __sub__(self, other): return self._sub(other)
    def __rsub__(self, other): return self._sub(other, True)

    def __neg__(self): return self._neg()

    def __mul__(self, other): return self._mul(other)
    def __rmul__(self, other): return self._mul(other)
    
    def __truediv__(self, other): return self._div(other)
    def __rtruediv__(self, other): return self._div(other, True)
    
    def __pow__(self, other): return self._pow(other)
    def __rpow__(self, other): return self._pow(other, True)
    
    def exp(self):
        out = Value(math.e ** self.data, (self,))
        return out

    def sigmoid(self):
        out = Value(1 / (1 + (-self).exp().data), (self,))
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,))
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
i = a.exp()
j = a.sigmoid()
k = a.relu()
print(f'{i}\n{j}\n{k}')
