from __future__ import annotations
from typing import List, Tuple

from random import gauss

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
        return f"Value({self.data})"

class Matrix:
    class Shape:
        def __init__(self, row: int, col: int) -> None:
            assert row > 0 and col > 0, "Row and column must be natural numbers."
            self.row = row
            self.col = col
        
        def __eq__(self, __value: Matrix.Shape) -> bool:
            return self.row == __value.row and self.col == __value.col
            
    _empty_list_error_message = "Cannot construct Matrix from empty list."
    _row_len_error_message = "Cannot construct Matrix. All rows must have the same length."
    # TODO: Update this error message based on whether scalar, 1d or 2d array was used.
    _type_error_message = "Cannot construct Matrix with given arguments. Expected: float, list of floats or list of lists of floats."
    
    # NOTE: Mirko A. (11/23/2023) 
    # Please do not use the constructor directly outside of the Matrix class.
    # Matrix can be constructed through the following static methods:
    # 1) from_scalar(float)
    # 2) from_1d_array(List[float])
    # 3) from_2d_array(List[List[float]])
    def __init__(self, data: List[List[Value]]) -> None:
        assert all((isinstance(x, Value) for x in row) for row in data), "Cannot construct Matrix. Must pass a 2D list of Value objects."
        self.data = data
        self._shape = Matrix.Shape(len(data), len(data[0]))
        
    @staticmethod
    def from_scalar(data: float) -> Matrix:
        if not isinstance(data, float):
            raise TypeError(Matrix._type_error_message)
        
        return Matrix([[Value(data)]])
    
    @staticmethod
    def from_1d_array(data: List[float]) -> Matrix:
        if not data: 
            raise ValueError(Matrix._empty_list_error_message)
        if not all(isinstance(float, x) for x in data):
            raise TypeError(Matrix._type_error_message)
        
        _data = []

        for x in data:
            _data.append(Value(x))

        return Matrix([_data])
    
    @staticmethod
    def from_2d_array(data: float) -> Matrix:
        if not all(row for row in data):
            raise ValueError(Matrix._empty_list_error_message)
        elif not all(len(row) == len(data[0]) for row in data):
            raise ValueError(Matrix._row_len_error_message)
        elif not all((isinstance(x, float) for x in row) for row in data):
            raise TypeError(Matrix._type_error_message)
        
        _data = []

        for row in data:
            _row = []
            for x in row:
                _row.append(Value(x))
            _data.append(_row)

        return Matrix(_data)
    
    # ------------------ Operators ------------------

    def add(self, other: Matrix) -> Matrix:
        assert isinstance(other, Matrix), f"Cannot add Matrix and {type(other)}."
        assert self.dims_match_with(other), "Cannot add Matrices if size doesn't match."

        rows, cols = self.shape.row, self.shape.col
        
        out_data = []
        #= Matrix.zeros(rows, cols)
        
        for row in range(rows):
            out_row = []
            for col in range(cols):
                out_row.append(self.data[row][col] + other.data[row][col])
            out_data.append(out_row)

        return Matrix(out_data)
    
    def T(self) -> Matrix:
        rows, cols = self.shape.row, self.shape.col
        out_data = []

        for col_idx in range(cols):
            out_row = []
            for row_idx in range(rows):
                out_row.append(self.data[row_idx][col_idx])
            out_data.append(out_row)

        return Matrix(out_data)
    
    # ------------------ Dunder ops ------------------

    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        return self.add(other)

    # -------------------- Utility --------------------
    
    @property
    def shape(self) -> Shape:
        return self._shape
    
    def dims_match_with(self, other: Matrix) -> bool:
        return self.shape == other.shape

    def __repr__(self) -> str:
        repr = str("Matrix([")

        for row in self.data:
            if not row == self.data[0]:
                repr += "        ["
            else:
                repr += "["
            
            for value in row:
                value_str = f"{value.data:.4f}"
                if value.data > 0:
                    # Indent to align with '-' character of negative numbers
                    value_str = " " + value_str
                    
                if not value == row[-1]:
                    repr += value_str + ", "
                else:
                    repr += value_str

            if not row == self.data[-1]:
                repr += "],\n"
            else:
                repr += "]"

        repr += "])"

        return repr
    
    # ----------------- Static methods ------------------

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        return Matrix.from_2d_array([ [0.0] * cols for _ in range(rows)])

    @staticmethod
    def randn(rows: int, cols: int, mean: float = 0.0, std_dev: float = 1.0) -> Matrix:
        data = [[gauss(mean, std_dev) for _ in range(rows)] for _ in range(cols)]
        return Matrix.from_2d_array(data)

m = Matrix.from_2d_array([[0.7,  2.1], [0.2,  4.1],  [2.3, 1.7]])
n = Matrix.from_2d_array([[1.3, -0.1], [1.8, -2.1], [-0.3, 0.3]])
y = Matrix.randn(2, 3)
y_t = y.T()
print(m + n)
print(y)
print(y_t)

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
