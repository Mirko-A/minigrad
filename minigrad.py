from __future__ import annotations

from random import gauss

import math

# import torch

# NOTE: Children of a Value which is created as a result of commutative operations
#       might be swapped if it is called from r-operation. Potentailly the reverse
#       flag might be needed for those operations just to keep the children always
#       in the same order.

class Value:
    def __init__(self, data: float, _children: tuple[Value, Value]=()) -> None:
        self.grad = 0.0
        self._data = data
        self._children = _children
        self._backward = lambda : None

    @property
    def data(self) -> float:
        return self._data

    # Operations

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
            y.grad += out.grad * ((x.data ** y.data) * math.log(x.data))

        out._backward = _backward

        return out

    def exp(self) -> Value:
        return math.e ** self

    # Operator magic methods

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

    # Activation funcs

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
        
    # Backpropagation

    def backward(self, grad = 1.0) -> None:
        self.grad = grad

        nodes: list[Value] = []
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

    # Utility

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
            
    # Matrix construction error messages
    _empty_list_error_message = "Cannot construct Matrix from empty list."
    _row_len_error_message = "Cannot construct Matrix. All rows must have the same length."
    _type_error_message = "Cannot construct Matrix with given arguments. Expected: "
    
    # NOTE: Mirko A. (11/23/2023) 
    # Please do not use the constructor directly outside of the Matrix class.
    # Matrix can be constructed through the following static methods:
    # 1) from_scalar(float)
    # 2) from_1d_array(list[float])
    # 3) from_2d_array(list[list[float]])
    def __init__(self, data: list[list[Value]]) -> None:
        assert all((isinstance(x, Value) for x in row) for row in data), "Cannot construct Matrix. Must pass a 2D list of Value objects."
        self.data = data
        self._shape = Matrix.Shape(len(data), len(data[0]))
        
    # Static construction methods

    @staticmethod
    def from_scalar(data: float) -> Matrix:
        if not isinstance(data, float):
            raise TypeError(Matrix._type_error_message + "scalar (float).")
        
        return Matrix([[Value(data)]])
    
    @staticmethod
    def from_1d_array(data: list[float]) -> Matrix:
        if not data: 
            raise ValueError(Matrix._empty_list_error_message)
        if not all(isinstance(x, float) for x in data):
            raise TypeError(Matrix._type_error_message + "1D array (float).")
        
        _data = []

        for x in data:
            _data.append(Value(x))

        return Matrix([_data])
    
    @staticmethod
    def from_2d_array(data: list[list[float]]) -> Matrix:
        if not all(row for row in data):
            raise ValueError(Matrix._empty_list_error_message)
        elif not all(len(row) == len(data[0]) for row in data):
            raise ValueError(Matrix._row_len_error_message)
        elif not all(all(isinstance(x, float) for x in row) for row in data):
            raise TypeError(Matrix._type_error_message + "2D array (float).")
        
        _data = []

        for row in data:
            _row = []
            for x in row:
                _row.append(Value(x))
            _data.append(_row)

        return Matrix(_data)
    
    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        return Matrix.from_2d_array([ [0.0] * cols for _ in range(rows)])

    @staticmethod
    def randn(rows: int, cols: int, mean: float = 0.0, std_dev: float = 1.0) -> Matrix:
        data = [[gauss(mean, std_dev) for _ in range(rows)] for _ in range(cols)]
        return Matrix.from_2d_array(data)
    
    # Operations

    def add(self, other: Matrix) -> Matrix:
        assert isinstance(other, Matrix), f"Cannot add Matrix and {type(other)}."
        assert self._dims_match_with(other), "Cannot add Matrices if shape doesn't match."

        rows, cols = self.shape.row, self.shape.col
        out_data = []
        
        for row in range(rows):
            out_row = []
            for col in range(cols):
                out_row.append(self.data[row][col] + other.data[row][col])
            out_data.append(out_row)

        return Matrix(out_data)
    
    def mul(self, other: float) -> Matrix:
        assert isinstance(other, float), f"Cannot multiply Matrix and {type(other)}."

        out_data = []

        for row in self.data:
            out_row = []
            for col in row:
                out_row.append(col * other)
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

    def flatten(self) -> Matrix:
        out_data = []

        for row in self.data:
            for value in row:
                out_data.append(value)

        return Matrix([out_data])

    def sum(self, dim: int | None = None) -> Matrix:
        VALID_AXIS_VALUES = [None, 0, 1]
        assert dim in VALID_AXIS_VALUES, "Invalid value for dim provided. Expected: None, 0 or 1."
        
        def sum_all(in_mat: Matrix) -> Matrix:
            out_data = Value(0.0)

            for row in in_mat.data:
                for value in row:
                    out_data += value

            return Matrix([[out_data]])

        def sum_along_dim(in_mat: Matrix, dim: int) -> Matrix:
            if dim == 0:
                in_mat = in_mat.T()

            out_data = []

            for row in in_mat.data:
                out_row = Value(0.0)
                
                for value in row:
                    out_row += value

                out_data.append([out_row])

            return Matrix(out_data)

        if dim is None:
            return sum_all(self)
        else:
            return sum_along_dim(self, dim)

    def exp(self) -> Matrix:
        out_data = []

        for row in self.data:
            out_row = []

            for value in row:
                out_row.append(value.exp())

            out_data.append(out_row)

        return Matrix(out_data)

    # Static operations

    @staticmethod
    def matmul(x: Matrix, y: Matrix) -> Matrix:
        assert isinstance(x, Matrix), f"Invalid type for matrix matmul product: {type(x)}."
        assert isinstance(y, Matrix), f"Invalid type for matrix matmul product: {type(y)}."
        assert x._inner_dims_match_with(y), f"Cannot multiply {x.shape.row}x{x.shape.col} and {y.shape.row}x{y.shape.col} Matrices. Inner dimensions must match."
    
        x_rows, y_rows, y_cols = x.shape.row, y.shape.row, y.shape.col
        out_data = []
        
        for x_row in range(x_rows):
            out_row = []
            for y_col in range(y_cols):
                temp_data = 0
                for y_row in range(y_rows):
                    temp_data += x.data[x_row][y_row] * y.data[y_row][y_col]
                out_row.append(temp_data)
            out_data.append(out_row)
                
        return Matrix(out_data)
                
    @staticmethod
    def are_equal(x: Matrix, y: Matrix) -> bool:
        assert isinstance(x, Matrix), f"Invalid type for matrix equality: {type(x)}."
        assert isinstance(y, Matrix), f"Invalid type for matrix equality: {type(y)}."
        assert x._dims_match_with(y), "Cannot compare Matrices if shape doesn't match."

        rows, cols = x.shape.row, x.shape.col

        for row in range(rows):
            for col in range(cols):
                if x[row][col] != y[row][col]:
                    return False

        return True
    
    # Operator magic methods

    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        return self.add(other)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __rmul__(self, other):
        return self.mul(other)
    
    def __truediv__(self, other):
        return self.mul(1.0/other)
    
    def __eq__(self, other):
        return Matrix.are_equal(self, other)
    
    def __getitem__(self, key):
        return self.data[key]

    # Activation funcs
    
    def sigmoid(self):
        rows, cols = self.shape.row, self.shape.col
        out_data = []
        
        for row in range(rows):
            out_row = []
            for col in range(cols):
                out_row.append(self.data[row][col].sigmoid())
            out_data.append(out_row)

        return Matrix(out_data)

    def relu(self):
        rows, cols = self.shape.row, self.shape.col
        out_data = []
        
        for row in range(rows):
            out_row = []
            for col in range(cols):
                out_row.append(self.data[row][col].relu())
            out_data.append(out_row)

        return Matrix(out_data)

    def tanh(self):
        rows, cols = self.shape.row, self.shape.col
        out_data = []
        
        for row in range(rows):
            out_row = []
            for col in range(cols):
                out_row.append(self.data[row][col].tanh())
            out_data.append(out_row)

        return Matrix(out_data)
    
    def softmax(self, axis=0):
        matrix = self if axis == 1 else self.T()
        rows, cols = matrix.shape.row, matrix.shape.col
        expsum = matrix._expsum()
        out_data = []
        
        for row in range(rows):
            out_row = []
            for col in range(cols):
                sm = matrix.data[row][col].exp() / expsum[row]
                out_row.append(sm)
            out_data.append(out_row)

        return Matrix(out_data).T() if axis == 0 else Matrix(out_data)

    # Backpropagation

    def backward(self) -> None:
        ...

    # Utility
    
    @property
    def shape(self) -> Shape:
        return self._shape
    
    def _expsum(self) -> list:
        rows, cols = self.shape.row, self.shape.col
        out_data = []
        
        for row in range(rows):
            row_expsum = 0.0
            for col in range(cols):
                val_exp = self.data[row][col].exp()
                row_expsum += val_exp
            out_data.append(row_expsum)
            
        return out_data
    
    def _dims_match_with(self, other: Matrix) -> bool:
        return self.shape == other.shape
    
    def _inner_dims_match_with(self, other: Matrix) -> bool:
        return self.shape.col == other.shape.row

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

m = Matrix.from_2d_array([[0.7,  2.1], [0.2,  4.1],  [2.3, 1.7]])
n = Matrix.from_2d_array([[1.8, -2.1], [-0.3, 0.3]])

#m_pt = torch.tensor([[0.7,  2.1], [0.2,  4.1],  [2.3, 1.7]]); m_pt.requires_grad = True
#n_pt = torch.tensor([[1.8, -2.1], [-0.3, 0.3]]); m_pt.requires_grad = True

# sigmoid = m.sigmoid()
# relu = m.relu()
# tanh = m.tanh()
softmax_0 = m.softmax(0)
softmax_1 = m.softmax(1)
# print(sigmoid)
# print(relu)
# print(tanh)
print(f'Original:\n{m}\n\nSoftmax axis 0:\n{softmax_0}\n\nSoftmax axis 1:\n{softmax_1}')

f = Matrix.matmul((2.0*m), n)
#f_t = torch.matmul((2.0*m_pt), n_pt)

#o = torch.tensor([2.0, 1.7, 3.3])
#o = o.softmax(0)

#f_t = f_t.softmax(1)
#print(f_t)

#f.backward()
#f_t.backward()

# for row in f.data:
#     for value in row:
#         print(value.grad)

# for row in f_t.data:
#     for value in row:
#         print(value.grad)