from __future__ import annotations
from enum import Enum
from random import gauss, uniform

import math

from minitorch.value import Value

class Matrix:
    class Diagonal(Enum):
        MAIN = 0
        ANTI = 1

    class Shape:
        def __init__(self, row: int, col: int) -> None:
            assert row > 0 and col > 0, "Row and column must be natural numbers."
            
            self.row = row
            self.col = col
        
        def __eq__(self, __value) -> bool:
            assert isinstance(__value, Matrix.Shape), f"Cannot compare a Shape and a {type(__value)}"
            return self.row == __value.row and self.col == __value.col
            
    # Matrix construction error messages
    _empty_list_error_message = "Cannot construct Matrix from empty list."
    _row_len_error_message = "Cannot construct Matrix. All rows must have the same length."
    _type_error_message = "Cannot construct Matrix with given arguments. Expected: "
    
    # Valid dimension values for a Matrix
    _VALID_DIM_VALUES = [0, 1]

    # NOTE: Mirko A. (11/23/2023) 
    # Please do not use the constructor directly outside of the Matrix class.
    # Matrix can be constructed through the following static methods:
    # 1) from_scalar(float)
    # 2) from_1d_array(list[float])
    # 3) from_2d_array(list[list[float]])
    def __init__(self, data: list[list[Value]], requires_grad: bool = True) -> None:
        assert all(all(isinstance(x, Value) for x in row) for row in data), "Cannot construct Matrix. Must pass a 2D list of Value objects."
        
        self.data = data
        self.requires_grad = requires_grad
        self._shape = Matrix.Shape(len(data), len(data[0]))

    # Static construction methods

    @staticmethod
    def from_scalar(data: float, requires_grad: bool = True) -> Matrix:
        assert isinstance(data, (float, int)), Matrix._type_error_message + "scalar (float or int)."
        
        return Matrix([[Value(data, requires_grad)]], requires_grad)
    
    @staticmethod
    def from_1d_array(data: list[float], requires_grad: bool = True) -> Matrix:
        assert isinstance(data, list) and                          \
               all(isinstance(value, (float, int)) for value in data),    \
               Matrix._type_error_message + "1D array (float or int)."
        assert data, Matrix._empty_list_error_message
        
        out_data = []

        for x in data:
            out_data.append(Value(x, requires_grad))

        return Matrix([out_data], requires_grad)
    
    @staticmethod
    def from_2d_array(data: list[list[float]], requires_grad: bool = True) -> Matrix:
        assert isinstance(data, list) and                                     \
               all(isinstance(row, list) for row in data) and                 \
               all(all(isinstance(x, (float, int)) for x in row) for row in data),   \
               Matrix._type_error_message + "2D array (float or int)."
        assert all(row for row in data), Matrix._empty_list_error_message
        assert all(len(row) == len(data[0]) for row in data), Matrix._row_len_error_message

        out_data = []

        for row in data:
            value_row = []

            for x in row:
                value_row.append(Value(x, requires_grad))

            out_data.append(value_row)

        return Matrix(out_data, requires_grad)
    
    @staticmethod
    def cat(matrices: list[Matrix], dim: int = 0) -> Matrix:
        assert dim in Matrix._VALID_DIM_VALUES, "Invalid dimension value provided. Expected: 0 or 1."
        assert all(isinstance(m, Matrix) for m in matrices), f"Cannot concatenate Matrix with other data types."

        def cat_rows(matrices: list[Matrix]) -> Matrix:
            rows = matrices[0].shape.row
            requires_grad = matrices[0].requires_grad
            assert all(m.shape.row == rows and m.requires_grad == requires_grad for m in matrices)

            out_data = []

            for row in range(rows):
                out_row = sum((m[row] for m in matrices), [])
                out_data.append(out_row)

            return Matrix(out_data, requires_grad)

        def cat_cols(matrices: list[Matrix]) -> Matrix:
            cols = matrices[0].shape.col 
            requires_grad = matrices[0].requires_grad
            assert all(m.shape.row == cols and m.requires_grad == requires_grad for m in matrices)

            out_data = [row for m in matrices for row in m.data]
            return Matrix(out_data, requires_grad)

        if dim == 0:
            return cat_cols(matrices)
        else:
            return cat_rows(matrices)

    # Static Matrix generation methods

    @staticmethod
    def fill(rows: int, cols: int, value: float, requires_grad: bool = True) -> Matrix:
        return Matrix.from_2d_array([[value] * cols for _ in range(rows)], requires_grad)

    @staticmethod
    def zeros(rows: int, cols: int, requires_grad: bool = True) -> Matrix:
        return Matrix.fill(rows, cols, 0.0, requires_grad)

    @staticmethod
    def one_hot(hot_index: int, num_classes: int, requires_grad: bool = True) -> Matrix:
        mask = [False]*num_classes
        mask[hot_index] = True
        return Matrix.masked_fill(Matrix.zeros(1, num_classes, requires_grad), [mask], 1.0)

    @staticmethod
    def randn(rows: int, cols: int, mean: float = 0.0, std_dev: float = 1.0, requires_grad: bool = True) -> Matrix:
        data = [[gauss(mean, std_dev) for _ in range(cols)] for _ in range(rows)]
        return Matrix.from_2d_array(data, requires_grad)
    
    @staticmethod
    def uniform(rows: int, cols: int, low: float, high: float, requires_grad: bool = True) -> Matrix:
        data = [[uniform(low, high) for _ in range(cols)] for _ in range(rows)]
        return Matrix.from_2d_array(data, requires_grad)

    @staticmethod
    def masked_fill(input: Matrix, mask: list[list[bool]], new_value: float) -> Matrix:
        assert len(mask) == input.shape.row and                              \
               all(len(mask_row) == input.shape.col for mask_row in mask),   \
               "Input Matrix and mask must have the same dimensions."
        
        out_data = []

        for in_data_row, mask_row in zip(input.data, mask):
            out_data_row = []

            for in_data_value, mask_value in zip(in_data_row, mask_row):
                out_data_row.append(Value(new_value, input.requires_grad) if mask_value == True else in_data_value)

            out_data.append(out_data_row)

        return Matrix(out_data, input.requires_grad)
        
    @staticmethod
    def replace(input: Matrix, target: float, new: float) -> Matrix:
        out_data = []
        
        for row in input.data:
            out_row = []

            for value in row:
                out_row.append(Value(new, input.requires_grad) if value.data == target else value)

            out_data.append(out_row)

        return Matrix(out_data, input.requires_grad)
    
    @staticmethod
    def tril(input: Matrix, diagonal: Diagonal = Diagonal.MAIN) -> Matrix:
        assert input._is_square(), "Cannot apply tril to non-square matrices."
        
        def tril_main_diagonal(input: Matrix) -> Matrix:
            out_data = []
            tril_cursor = 1

            for row in input.data:
                out_row = []

                for value_pos, value in enumerate(row):
                    should_keep_value = value_pos < tril_cursor
                    out_row.append(value if should_keep_value else Value(0.0, input.requires_grad))

                out_data.append(out_row)
                tril_cursor += 1
            
            return Matrix(out_data, input.requires_grad)
        
        def tril_anti_diagonal(input: Matrix) -> Matrix:
            out_data = []
            tril_cursor = 0

            for row in input.data:
                out_row = []

                for value_pos, value in enumerate(row):
                    should_replace_value = value_pos < tril_cursor
                    out_row.append(Value(0.0, input.requires_grad) if should_replace_value else value)

                out_data.append(out_row)
                tril_cursor += 1
            
            return Matrix(out_data, input.requires_grad)

        output = tril_main_diagonal(input) if diagonal == Matrix.Diagonal.MAIN else tril_anti_diagonal(input)

        return output

    # Shape manipulation methods

    def T(self) -> Matrix:
        rows, cols = self.shape.row, self.shape.col
        out_data = []

        for col_idx in range(cols):
            out_row = []

            for row_idx in range(rows):
                out_row.append(self.data[row_idx][col_idx])

            out_data.append(out_row)
            
        return Matrix(out_data, self.requires_grad)

    def flatten(self) -> Matrix:
        out_data = []

        for row in self.data:
            for value in row:
                out_data.append(value)

        return Matrix([out_data], self.requires_grad)

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

        x, y = self, other
        out_data = []
        
        for x_row, y_row in zip(x.data, y.data):
            out_row = []

            for x_value, y_value in zip(x_row, y_row):
                out_row.append(x_value + y_value)

            out_data.append(out_row)

        return Matrix(out_data, x.requires_grad or y.requires_grad)
    
    def mul(self, other: float | int) -> Matrix:
        assert isinstance(other, (float | int)), f"Cannot multiply Matrix and {type(other)}."

        x, y = self, other
        out_data = []

        for row in x.data:
            out_row = []

            for value in row:
                out_row.append(value * y)

            out_data.append(out_row)

        return Matrix(out_data, x.requires_grad)

    def sum(self, dim: int | None = None) -> Matrix:
        assert dim in Matrix._VALID_DIM_VALUES + [None], "Invalid dimension value provided. Expected: None, 0 or 1."
        
        def sum_all(input: Matrix) -> Matrix:
            out_data = 0

            for row in input.data:
                for value in row:
                    out_data += value

            return Matrix([[out_data]], input.requires_grad)

        def sum_along_dim(input: Matrix, dim: int) -> Matrix:
            input = input if dim == 1 else input.T()
            out_data = []

            for row in input.data:
                out_row = 0
                
                for value in row:
                    out_row += value

                out_data.append([out_row])

            return Matrix(out_data, input.requires_grad)

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

        return Matrix(out_data, self.requires_grad)
    
    def log(self, base: Value | float | int = math.e) -> Matrix:
        out_data = []

        for row in self.data:
            out_row = []

            for value in row:
                out_row.append(value.log(base))

            out_data.append(out_row)

        return Matrix(out_data, self.requires_grad)

    def matmul(self, other: Matrix) -> Matrix:
        assert isinstance(other, Matrix), f"Invalid type for matrix matmul product: {type(other)}."
        assert self._inner_dims_match_with(other), \
               f"Cannot multiply {self.shape.row}x{self.shape.col} and {other.shape.row}x{other.shape.col} Matrices. Inner dimensions must match."
    
        x, y = self, other.T()
        out_data = []
        
        for x_row in x.data:
            out_row = []

            for y_row in y.data:
                out_value = 0

                for x_value, y_value in zip(x_row, y_row):
                    out_value += x_value * y_value

                out_row.append(out_value)

            out_data.append(out_row)
                
        return Matrix(out_data, self.requires_grad or other.requires_grad)

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

    # Activation functions
    
    def sigmoid(self):
        out_data = []
        
        for row in self.data:
            out_row = []

            for value in row:
                out_row.append(value.sigmoid())

            out_data.append(out_row)

        return Matrix(out_data, self.requires_grad)

    def relu(self):
        out_data = []
        
        for row in self.data:
            out_row = []

            for value in row:
                out_row.append(value.relu())

            out_data.append(out_row)

        return Matrix(out_data, self.requires_grad)

    def tanh(self):
        out_data = []
        
        for row in self.data:
            out_row = []

            for value in row:
                out_row.append(value.tanh())

            out_data.append(out_row)

        return Matrix(out_data, self.requires_grad)
    
    def softmax(self, dim: int = 0):
        assert dim in Matrix._VALID_DIM_VALUES, "Invalid dimension value provided. Expected: 0 or 1."

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

        out_mat = Matrix(out_data, self.requires_grad)

        return out_mat if dim == 1 else out_mat.T()

    # Cost functions
    
    def cross_entropy(self, target: Matrix):
        assert isinstance(target, Matrix), f"Cannot perform Cross-Entropy on target type {type(target)}"
        assert self._dims_match_with(target), "Cannot perform Cross-Entropy. Dimensions of input don't match with target."
        # NOTE: PyTorch uses base e here, might be relevant later
        input_log = self.log(2)
        out_data = []
        
        for target_row, input_log_row in zip(target.data, input_log.data):
            cross_entropy_sum = 0
            
            for target_value, input_log_value in zip(target_row, input_log_row):
                cross_entropy = target_value * input_log_value
                cross_entropy_sum += cross_entropy
                
            out_data.append(-cross_entropy_sum)
            
        return Matrix([out_data], self.requires_grad)
    
    def MSE(self, target: Matrix):
        assert isinstance(target, Matrix), f"Cannot perform MSE on target type {type(target)}"
        assert self._dims_match_with(target), "Cannot perform MSE. Dimensions of input don't match with target."
        
        MSE = []
        
        for input_row, target_row in zip(self.data, target.data):
            row_error_sum = 0
            
            for input_value, target_value in zip(input_row, target_row):
                squared_error = (target_value - input_value) ** 2
                row_error_sum += squared_error
                
            MSE.append(row_error_sum / self.shape.col)
        
        return Matrix([MSE], self.requires_grad)

    # Backpropagation

    def backward(self, grad: float = 1.0) -> None:
        data = self.item()
        assert isinstance(data, Value), f"Cannot call backward on {type(data)}. Expected scalar."
        data.backward(grad)

    # Utility
    
    @property
    def shape(self) -> Shape:
        return self._shape
    
    # Function returns one of the following:
    # a) list[list[Value]] -> When row >  1 and col >  1 
    # b) list[Value]       -> When row == 1 and col >  1
    # c) Value             -> When row == 1 and col == 1
    def item(self) -> list[list[Value]] | list[Value] | Value:
        row, col = self.shape.row, self.shape.col

        if row > 1 and col > 1:
            out_data: list[list[Value]] | list[Value] | Value = self.data
        elif row > 1 and col == 1:
            out_data = [row[0] for row in self.data]
        elif row == 1 and col > 1:
            out_data = self.data[0]
        else:
            out_data = self.data[0][0]

        return out_data

    def grad(self) -> list[list[float]]:
        return [[data.grad for data in row] for row in self.data]

    def _dims_match_with(self, other: Matrix) -> bool:
        return self.shape == other.shape
    
    def _inner_dims_match_with(self, other: Matrix) -> bool:
        return self.shape.col == other.shape.row
    
    def _is_square(self) -> bool:
        return self.shape.row == self.shape.col

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
