from __future__ import annotations
from typing import List, Tuple

import math

#import torch

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
        return f"{self.data}"

class Matrix:
    def __init__(self, data: float | List[float] | List[List[float]]) -> None:
        self.empty_list_error_message = "Cannot construct Matrix from empty list."
        self.row_len_error_message = "All rows must have the same length."
        self.type_error_message = "Cannot construct Matrix with given arguments. Expected: float, list of floats or list of lists of floats."
        
        if isinstance(data, float):
            self.data = self._create_data_from_float(data)
        elif isinstance(data, list) and all(isinstance(x, float) for x in data):
            self.data = self._create_data_from_list_1d(data)
        elif isinstance(data, list) and all(isinstance(row, list) and all(isinstance(x, float) for x in row) for row in data):
            self.data = self._create_data_from_list_2d(data)
        else:
            raise TypeError(self.type_error_message)

    def _create_data_from_float(self, data: float) -> List[List[Value]]:
        return [[Value(data)]]
    
    def _create_data_from_list_1d(self, data: List[float]) -> List[List[Value]]:
        if not data: 
            raise ValueError(self.empty_list_error_message)
        
        for index, value in enumerate(data):
            data[index] = Value(value)

        return [data]

    def _create_data_from_list_2d(self, data: List[List[float]]) -> List[List[Value]]:
        if not all(row for row in data):
            raise ValueError(self.empty_list_error_message)
        elif not all(len(row) == len(data[0]) for row in data):
            raise ValueError(self.row_len_error_message)            

        for row in data:
            for index, value in enumerate(row):
                row[index] = Value(value)
            
        return data
    
    # -------------------- Utility --------------------
    def __repr__(self) -> str:
        repr = str("Matrix([")

        for row in self.data:
            if not row == self.data[0]:
                repr += "        ["
            else:
                repr += "["
            
            for value in row:
                if not value == row[-1]:
                    repr += f"{value}, "
                else:
                    repr += f"{value}"

            if not row == self.data[-1]:
                repr += "],\n"
            else:
                repr += "]"

        repr += "])"

        return repr
    
m = Matrix([[0.7, 2.1], [0.2, 4.1], [2.3, 1.7]])
print(m)

a = Value(0.31)
b = Value(1.5)

#t = torch.tensor([[0.7, 2.1], [0.2, 4.1], [2.3, 1.7]])
#print(t)
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
