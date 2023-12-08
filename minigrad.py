from __future__ import annotations

from enum import Enum
from random import gauss

import math

# NOTE: Children of a Value which is created as a result of commutative operations
#       might be swapped if it is called from r-operation. Potentailly the reverse
#       flag might be needed for those operations just to keep the children always
#       in the same order.

class Value:
    def __init__(self, data: float, _children: tuple[Value, Value]=()) -> None:
        assert isinstance(data, float), f"Cannot construct Value object with type: {type(data)}. Expected float."

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
    class Diagonal(Enum):
        LOWER = 0
        UPPER = 1

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
        assert all(all(isinstance(x, Value) for x in row) for row in data), "Cannot construct Matrix. Must pass a 2D list of Value objects."
        
        self.data = data
        self._shape = Matrix.Shape(len(data), len(data[0]))
        
    # Static construction methods

    @staticmethod
    def from_scalar(data: float) -> Matrix:
        assert isinstance(data, float), Matrix._type_error_message + "scalar (float)."
        
        return Matrix([[Value(data)]])
    
    @staticmethod
    def from_1d_array(data: list[float]) -> Matrix:
        assert isinstance(data, list) and                          \
               all(isinstance(value, float) for value in data),    \
               Matrix._type_error_message + "1D array (float)."
        assert data, Matrix._empty_list_error_message
        
        _data = []

        for x in data:
            _data.append(Value(x))

        return Matrix([_data])
    
    @staticmethod
    def from_2d_array(data: list[list[float]]) -> Matrix:
        assert isinstance(data, list) and                                     \
               all(isinstance(row, list) for row in data) and                 \
               all(all(isinstance(x, float) for x in row) for row in data),   \
               Matrix._type_error_message + "2D array (float)."
        assert all(row for row in data), Matrix._empty_list_error_message
        assert all(len(row) == len(data[0]) for row in data), Matrix._row_len_error_message

        value_data = []

        for row in data:
            value_row = []

            for x in row:
                value_row.append(Value(x))

            value_data.append(value_row)

        return Matrix(value_data)
    
    # Static Matrix generation methods

    @staticmethod
    def fill(rows: int, cols: int, value: float) -> Matrix:
        return Matrix.from_2d_array([[value] * cols for _ in range(rows)])

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        return Matrix.fill(rows, cols, 0.0)

    @staticmethod
    def randn(rows: int, cols: int, mean: float = 0.0, std_dev: float = 1.0) -> Matrix:
        data = [[gauss(mean, std_dev) for _ in range(cols)] for _ in range(rows)]
        return Matrix.from_2d_array(data)
    
    @staticmethod
    def masked_fill(input: Matrix, mask: list[list[bool]], new_value: float) -> Matrix:
        assert len(mask) == input.shape.row and                              \
               all(len(mask_row) == input.shape.col for mask_row in mask),   \
               "Input Matrix and mask must have the same dimensions."
        
        out_data = []

        for in_data_row, mask_row in zip(input.data, mask):
            out_data_row = []

            for in_data_value, mask_value in zip(in_data_row, mask_row):
                out_data_row.append(Value(new_value) if mask_value == True else in_data_value)

            out_data.append(out_data_row)

        return Matrix(out_data)
        
    @staticmethod
    def replace(input: Matrix, target: float, new: float) -> Matrix:
        out_data = []
        
        for row in input.data:
            out_row = []

            for value in row:
                out_row.append(Value(new) if value.data == target else value)

            out_data.append(out_row)

        return Matrix(out_data)
    
    @staticmethod
    def tril(input: Matrix, diagonal: Diagonal = Diagonal.LOWER):
        assert input.shape.row == input.shape.col, "Can only apply tril to square matrices."
        out_data = []

        tril_cursor = 1

        for row in input.data:
            out_row = []

            for value_pos, value in enumerate(row):
                out_row.append(value if value_pos < tril_cursor else Value(0.0))

            out_data.append(out_row)
            tril_cursor += 1

        return Matrix(out_data)

    # Operations

    def is_equal_to(self, target: Matrix) -> bool:
        assert self._dims_match_with(target), "Cannot compare Matrices if shape doesn't match."

        rows, cols = self.shape.row, self.shape.col

        for row in range(rows):
            for col in range(cols):
                if self[row][col].data != target[row][col].data:
                    return False

        return True
    
    def is_elementwise_equal_to(self, target: float) -> list[list[bool]]:
        out_data = []

        for row in self.data:
            out_row = []

            for value in row:
                out_row.append(True if value.data == target else False)

            out_data.append(out_row)

        return out_data

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
        
        def sum_all(input: Matrix) -> Matrix:
            out_data = Value(0.0)

            for row in input.data:
                for value in row:
                    out_data += value

            return Matrix([[out_data]])

        def sum_along_dim(input: Matrix, dim: int) -> Matrix:
            input = input if dim == 1 else input.T()
            out_data = []

            for row in input.data:
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
        assert x._inner_dims_match_with(y), \
               f"Cannot multiply {x.shape.row}x{x.shape.col} and {y.shape.row}x{y.shape.col} Matrices. Inner dimensions must match."
    
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
        if isinstance(other, Matrix):
            return self.is_equal_to(other)
        elif isinstance(other, float):
            return self.is_elementwise_equal_to(other)
        else:
            assert False, f"Invalid type for matrix equality: {type(other)}. Expected Matrix or float."

    def __getitem__(self, key) -> list[Value]:
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
    
    def softmax(self, dim: int = 0):
        input = self if dim == 1 else self.T()
        input_exp = input.exp()
        input_exp_sums = input_exp.sum(dim=1).item()
        out_data = []
        
        for row_exp, row_exp_sum in zip(input_exp.data, input_exp_sums):
            out_row = []

            for value_exp in row_exp:
                probability = value_exp / row_exp_sum
                out_row.append(probability)
            out_data.append(out_row)

        output = Matrix(out_data)

        return output if dim == 1 else output.T()

    # Backpropagation

    def backward(self) -> None:
        # TODO: Naive implementation
        self.data[0][0].backward()

    # Utility
    
    @property
    def shape(self) -> Shape:
        return self._shape
    
    # Function returns one of the following:
    # a) list[list[Value]] -> When row >  1 and col > 1 
    # b) list[Value]       -> When row == 1 and col > 1
    # c) Value             -> When row == 1 and col == 1
    def item(self) -> list[list[Value]] | list[Value] | Value:
        row, col = self.shape.row, self.shape.col

        if row > 1 and col > 1:
            out_data = self.data
        elif row > 1 and col == 1:
            out_data = [row[0] for row in self.data]
        elif row == 1 and col > 1:
            out_data = self.data[0]
        else:
            out_data = self.data[0][0]

        return out_data

    # Returns gradients of all elements in a list[list[float]]
    def grad(self) -> list[list[float]]:
        return [[data.grad for data in row] for row in self.data]

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
