from __future__ import annotations
from typing import List, Tuple

import math

#import torch

# TODO: Begin Tensor class

# NOTE: Children of a Value which is created as a result of commutative operations
#       might be swapped if it is called from r-operation. Potentailly the reverse
#       flag might be needed for those operations just to keep the children always
#       in the same order.

class Value:
    def __init__(self, data: float, _children: Tuple[Value, Value]=()) -> None:
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

        out = Value(x.data - y.data, (x, y))

        def _backward():
            x.grad +=  out.grad
            y.grad += -out.grad

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

        out = Value(x.data / y.data, (x, y))

        def _backward():
            x.grad += out.grad * 1 / y.data
            y.grad += out.grad * (-1)*(x.data / (y.data ** 2))

        out._backward = _backward

        return out

    def _pow(self, other: Value | float, reverse=False) -> Value:
        if not isinstance(other, Value): other = Value(other)
        x, y = self, other
        
        if reverse:
            x, y = y, x

        out = Value(x.data ** y.data, (x, y))
        
        def _backward():
            x.grad += out.grad * (y.data * (x.data ** (y.data - 1)))
            y.grad += out.grad * ((x.data ** y.data) * math.log(y.data))

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
            return 1 / (1 + math.exp(-x))
        
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

        out._backward = _backward

        return out

    def tanh(self):
        def tanh_impl(x):
            return (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        
        out = Value(tanh_impl(self.data), (self,))
        
        def _backward():
            self.grad += out.grad * (1 - tanh_impl(self.data) ** 2)
        
        out._backward = _backward
        
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
    def exp(self) -> Value:
        return Value(math.exp(self.data))

    def __repr__(self) -> str:
        return f"Value: {self.data}"

a = Value(0.31)
b = Value(1.5)

#a_t = torch.Tensor([0.31]); a_t.requires_grad = True
#b_t = torch.Tensor([1.5]); b_t.requires_grad = True

f = ((a*0.7 + b) ** 2.0) / 1.7
f = f.tanh()
print(f)

#f_t = ((a_t*0.7 + b_t) ** 2.0) / 1.7
#f_t = f_t.tanh()
#print(f_t)

f.backward()
#f_t.backward()

grads_txt = f"""
a:{a.grad}
b:{b.grad}
"""

#grads_txt = f"""
#a:{a.grad}
#a_t:{a_t.grad.item()}
#b:{b.grad}
#b_t:{b_t.grad.item()}
#"""

print(grads_txt)
