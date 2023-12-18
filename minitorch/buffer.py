from __future__ import annotations
import math
from typing import Optional

class MiniBuffer:
    _VALID_DIM_VALUES = [0, 1]

    def __init__(self, data: list[list[float]]) -> None:
        assert all(len(row) == len(data[0]) for row in data), "Cannot create MiniBuffer. All rows must have the same length."
        self.data = data
        self.shape = (len(data), len(data[0]))

    # Static MiniBuffer generation operations

    @staticmethod
    def fill(rows: int, cols: int, value: float) -> MiniBuffer:
        out_data = []

        for _ in range(rows):
            out_row = []

            for _ in range (cols):
                out_row.append(value)

            out_data.append(out_row)

        return MiniBuffer(out_data)

    # Unary operations

    def neg(self) -> MiniBuffer:
        out_data = []

        for x_row in self.data:
            out_row = []

            for x_val in x_row:
                out_row.append(-x_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)

    def log(self, base: float = math.e) -> MiniBuffer:
        assert isinstance(base, float), f"Cannot perform log with non-scalar base. Expected: float, got {type(base)}"
        
        out_data = []

        for x_row in self.data:
            out_row = []

            for x_val in x_row:
                out_row.append(math.log(x_val, base))

            out_data.append(out_row)

        return MiniBuffer(out_data)

    # Reduce operations

    def sum(self, dim: Optional[int] = None):
        assert dim in MiniBuffer._VALID_DIM_VALUES + [None], "Invalid dimension value provided. Expected: None, 0 or 1."
        
        def sum_all(input: MiniBuffer) -> MiniBuffer:
            out_data = 0.0

            for row in input.data:
                for value in row:
                    out_data += value

            return MiniBuffer([[out_data]])

        def sum_along_dim(input: MiniBuffer, dim: int) -> MiniBuffer:
            input = input if dim == 1 else input.T()
            out_data = []

            for row in input.data:
                out_row = 0.0
                
                for value in row:
                    out_row += value

                out_data.append([out_row])

            return MiniBuffer(out_data)

        if dim is None:
            return sum_all(self)
        else:
            return sum_along_dim(self, dim)

    # Binary operations

    def add(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row, y_row in zip(self.data, other.data):
            out_row = []

            for x_val, y_val in zip(x_row, y_row):
                out_row.append(x_val + y_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)
    
    def sub(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row, y_row in zip(self.data, other.data):
            out_row = []

            for x_val, y_val in zip(x_row, y_row):
                out_row.append(x_val - y_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)

    def mul(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row, y_row in zip(self.data, other.data):
            out_row = []

            for x_val, y_val in zip(x_row, y_row):
                out_row.append(x_val * y_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)
    
    def div(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row, y_row in zip(self.data, other.data):
            out_row = []

            for x_val, y_val in zip(x_row, y_row):
                out_row.append(x_val / y_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)

    def pow(self, other: MiniBuffer) -> MiniBuffer:
        out_data = []

        for x_row, y_row in zip(self.data, other.data):
            out_row = []

            for x_val, y_val in zip(x_row, y_row):
                out_row.append(x_val ** y_val)

            out_data.append(out_row)

        return MiniBuffer(out_data)

    # Movemenet operations

    def flatten(self) -> MiniBuffer:
        rows, cols = self.shape[0], self.shape[1]
        out_data = []

        for col_idx in range(cols):
            out_row = []

            for row_idx in range(rows):
                out_row.append(self.data.data[row_idx][col_idx])

            out_data.append(out_row)
            
        return MiniBuffer(out_data)
    
    def T(self) -> MiniBuffer:
        rows, cols = self.shape[0], self.shape[1]
        out_data = []

        for col_idx in range(cols):
            out_row = []

            for row_idx in range(rows):
                out_row.append(self.data[row_idx][col_idx])

            out_data.append(out_row)
            
        return MiniBuffer(out_data)

    def expand(self, rows: int, cols: int) -> MiniBuffer:
        old_shape = self.shape
        out_data = []

        expand_along_rows = rows > old_shape[0]
        expand_along_cols = cols > old_shape[1]

        if expand_along_rows and expand_along_cols:
            assert MiniBuffer.is_scalar(), "Cannot expand a non-scalar along both dimensions."
            return MiniBuffer.fill(rows, cols, self[0][0])
        elif expand_along_rows:
            for _ in range(rows):
                out_data.append(rows[0])
                
            return MiniBuffer(out_data)
        elif expand_along_cols:
            for row_idx in range(old_shape[0]):
                out_data.append([self.data[row_idx][0]] * cols)

            return MiniBuffer(out_data)
        else:
            return MiniBuffer(self.data)

    def reshape(self, rows: int, cols: int) -> MiniBuffer:
        flattened = self.flatten()
        assert flattened.shape[1] == (rows * cols), "Cannot reshape, new dimensions don't match the current shape."

        out_data = []

        for row in range(rows):
            out_data.append(flattened[0][row : (row + cols)])

        return MiniBuffer(out_data)
    
    # Unary operator magic methods

    def __neg__(self):
        return self.neg()

    # Binary operator magic methods

    def __add__(self, other):
        assert isinstance(other, MiniBuffer), f"Cannot perform addition with MiniBuffer and {type(other)}."

        return self.add(other)
    
    def __sub__(self, other):
        assert isinstance(other, MiniBuffer), f"Cannot perform subtraction with MiniBuffer and {type(other)}."

        return self.sub(other)

    def __mul__(self, other):
        assert isinstance(other, MiniBuffer), f"Cannot perform multiplication with MiniBuffer and {type(other)}."

        return self.mul(other)

    def __truediv__(self, other):
        assert isinstance(other, MiniBuffer), f"Cannot perform division with MiniBuffer and {type(other)}."

        return self.div(other)
    
    def __pow__(self, other):
        assert isinstance(other, MiniBuffer), f"Cannot perform exponentiation with MiniBuffer and {type(other)}."

        return self.pow(other)
    
    # Utility

    def is_scalar(self) -> bool:
        return self.shape[0] == 1 and self.shape[1] == 1

    def __getitem__(self, key) -> list[float]:
        return self.data[key]
    
    def __repr__(self) -> str:
        repr = str("([")

        for row_idx, row in enumerate(self.data):
            if not row_idx == 0:
                repr += "          ["
            else:
                repr += "["
            
            for col_idx, value in enumerate(row):
                value_str = f"{value:.4f}"
                
                if value > 0:
                    # Indent to align with '-' character of negative numbers
                    value_str = " " + value_str
                    
                if not col_idx == (len(row) - 1):
                    repr += value_str + ", "
                else:
                    repr += value_str

            if not row_idx == (len(self.data) - 1):
                repr += "],\n"
            else:
                repr += "]"

        repr += "])"

        return repr