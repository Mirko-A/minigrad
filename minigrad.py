from __future__ import annotations
from typing import List

import math

# TODO: Add the rest of tests for Value class (take into account non comutative functions)
#       1) Prki:  Sub, mul, relu, tanh
#       2) Mire:  add, div, pow, sigmoid
#       
# TODO: Begin Tensor class

class Value:
    def __init__(self, data: float, _children: tuple=()) -> None:
        self.grad = 0.0
        self._data = data
        self._children = _children
        self._backward = lambda : None

    @property
    def data(self) -> float:
        return self._data

    # -------------------- Operators --------------------
    def _add(self, other: Value | float) -> Value:
        if not isinstance(other, Value): other = Value(other)

        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def _sub(self, other: Value | float, reverse=False) -> Value:
        if not isinstance(other, Value): other = Value(other)
        x, y = self, other
        
        if reverse:
            x, y = y, x

        out = Value(x.data - y.data, (self, other))

        def _backward():
            if reverse:
                self.grad  += -out.grad
                other.grad +=  out.grad
            else:
                self.grad  +=  out.grad
                other.grad += -out.grad

        out._backward = _backward
        
        return out
    
    def _neg(self) -> Value:
        out = Value(-self.data)
        
        def _backward():
            self.grad += -out.grad
        out._backward = _backward

        return out
    
    def _mul(self, other: Value | float) -> Value:
        if not isinstance(other, Value): other = Value(other)

        out = Value(self.data * other.data, (self, other))
        
        def _backward():
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out
    
    def _div(self, other: Value | float, reverse=False) -> Value:
        if not isinstance(other, Value): other = Value(other)
        x, y = self, other
        
        if reverse:
            x, y = y, x

        out = Value(x.data / y.data, (self, other))

        def _backward():
            if reverse:
                # (other / self)
                self.grad  += out.grad * other.data
                other.grad += out.grad * 1 / self.data
            else:
                # (self / other)
                self.grad  += out.grad * 1 / other.data
                other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def _pow(self, other: Value | float, reverse=False) -> Value:
        if not isinstance(other, Value): other = Value(other)
        x, y = self, other
        
        if reverse:
            x, y = y, x

        out = Value(x.data ** y.data, (self, other))
        
        def _backward():
            if reverse:
                # (other ** self)
                self.grad  += out.grad * ((other.data ** self.data) * math.log(self.data))
                other.grad += out.grad * (self.data * (other.data ** (self.data - 1)))
            else:
                # (self ** other)
                self.grad  += out.grad * (other.data * (self.data ** (other.data - 1)))
                other.grad += out.grad * ((self.data ** other.data) * math.log(other.data))

        out._backward = _backward

        return out

    # -------------------- Dunder ops --------------------
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

    # ----------------- Activation funcs -----------------
    def sigmoid(self):
        def sigmoid_impl(x):
            return 1 / (1 + math.e ** (-x))
        
        out = Value(sigmoid_impl(self.data), (self,))

        def _backward():
            self.grad += out.grad * (sigmoid_impl(self.data) * (1 - sigmoid_impl(self.data)))

        out._backward = _backward

        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,))

        def _backward():
            relu_deriv = 1 if self.data > 0 else 0
            self.grad += out.grad * relu_deriv

        self._backward = _backward

        return out

    # ---------------- Backpropagation ----------------
    def backward(self) -> None:
        self.grad = 1.0

        nodes: List[Value] = []
        visited = set()
        def toposort(node: Value):
            if node not in visited:
                visited.add(node)

                for child in node._children:
                    toposort(child)

                nodes.append(node)

        toposort(self)

        for node in reversed(nodes):
            node._backward()

    # -------------------- Utility --------------------
    def exp(self) -> float:
        return self ** math.e

    def __repr__(self) -> str:
        return f"Value: {self.data}"
    
a = Value(0.31)
b = Value(1.5)

c = a + b
d = c * 0.9
e = 2.6 + d
f = 0.15 + e
g = b ** f
h = g ** 0.62
h._backward()
i = a.exp()
j = a.sigmoid()
k = a.relu()
print(f'{i}\n{j}\n{k}')
