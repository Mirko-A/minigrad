from __future__ import annotations
from typing import Optional
from enum import Enum, auto
import numpy as np
import math

class MiniBuffer:
    # TODO: Add more...
    class UnaryOp(Enum):
        NEG  = 0
        LOG  = auto()
        LOG2 = auto()

    class BinaryOp(Enum):
        ADD = 0
        SUB = auto()
        MUL = auto()
        DIV = auto()
        POW = auto()
        MAX = auto()

    class ReshapeOp(Enum):
        PAD = 0
        SHRINK = auto()

    class CmpOp(Enum):
        LT = 0
        EQ = auto()
        GT = auto()

    def __init__(self, 
                 data: list[float], 
                 shape: tuple[int, ...], 
                 strides: Optional[tuple[int, ...]] = None):
        assert isinstance(data, list) and all(isinstance(value, float) for value in data), \
                f"Cannot construct buffer. Expected data type is list[float] but got: {type(data)}."
        assert isinstance(shape, tuple) and all(isinstance(dim, int) for dim in shape), \
                f"Cannot construct buffer. Expected shape type is tuple[int, ...] but got {type(shape)}"
        
        self.data = data
        self.shape = shape

        if strides is None:
            self.strides = MiniBuffer.get_strides_from_shape(shape)
        else:
            self.strides = strides

    # Static MiniBuffer generation operations

    @staticmethod
    def np_load(data: list) -> MiniBuffer:
        _np = np.array(data)
        shape = ()

        for shape_n in _np.shape:
            shape += (shape_n,)
        
        return MiniBuffer(_np.reshape(-1).astype(np.float32).tolist(), shape)

    @staticmethod
    def fill(shape: tuple[int, ...], value: float | int) -> MiniBuffer:
        if isinstance(value, int):
            value = float(value)
            
        total_elements = math.prod(shape)

        return MiniBuffer([value] * total_elements, shape)

    @staticmethod
    def replace(input: MiniBuffer, target: float, new: float) -> MiniBuffer:
        out_data = []
        
        for val in input.data:
            if val == target:
                out_data.append(new)
            else:
                out_data.append(val)

        return MiniBuffer(out_data, input.shape)

    @staticmethod
    def full_like(input: MiniBuffer, value: float | int) -> MiniBuffer:
        if isinstance(value, int):
            value = float(value)

        return MiniBuffer.fill(input.shape, value)

    @staticmethod
    def masked_fill(input: MiniBuffer, mask: list[bool], value: float) -> MiniBuffer:
        return MiniBuffer._traverse_dims_and_masked_fill(0,
                                                         0,
                                                         mask,
                                                         value,
                                                         input)
    
    # Unary operations

    def neg(self) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, ),
                                                          MiniBuffer.UnaryOp.NEG,
                                                          self)

        return MiniBuffer(out_data, self.shape)

    def log(self) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, ),
                                                          MiniBuffer.UnaryOp.LOG,
                                                          self)

        return MiniBuffer(out_data, self.shape)

    def log2(self) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, ),
                                                          MiniBuffer.UnaryOp.LOG2,
                                                          self)

        return MiniBuffer(out_data, self.shape)

    # Reduce operations

    # TODO: Not implemented!!
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
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, 0),
                                                          MiniBuffer.BinaryOp.ADD,
                                                          self, other)

        return MiniBuffer(out_data, self.shape)
    
    def sub(self, other: MiniBuffer) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, 0),
                                                          MiniBuffer.BinaryOp.SUB,
                                                          self, other)

        return MiniBuffer(out_data, self.shape)

    def mul(self, other: MiniBuffer) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, 0),
                                                          MiniBuffer.BinaryOp.MUL,
                                                          self, other)
    
        return MiniBuffer(out_data, self.shape)
    
    def div(self, other: MiniBuffer) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, 0),
                                                          MiniBuffer.BinaryOp.DIV,
                                                          self, other)

        return MiniBuffer(out_data, self.shape)
    
    def pow(self, other: MiniBuffer) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, 0),
                                                          MiniBuffer.BinaryOp.POW,
                                                          self, other)

        return MiniBuffer(out_data, self.shape)
    
    def max(self, other: MiniBuffer) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_op(0,
                                                          (0, 0),
                                                          MiniBuffer.BinaryOp.MAX,
                                                          self, other)

        return MiniBuffer(out_data, self.shape)
    
    # TODO: Equality functions are not working

    def is_equal_to(self, target: MiniBuffer) -> bool:
        return MiniBuffer._traverse_dims_and_compare(0,
                                                     (0, 0),
                                                     self,
                                                     target)

    def is_elementwise_greater_than(self, target: float) -> list[bool]:
        return MiniBuffer._traverse_dims_and_compare_elementwise(0,
                                                                 0,
                                                                 MiniBuffer.CmpOp.GT,
                                                                 self,
                                                                 target)
    
    def is_elementwise_less_than(self, target: float) -> list[bool]:
        return MiniBuffer._traverse_dims_and_compare_elementwise(0,
                                                                 0,
                                                                 MiniBuffer.CmpOp.LT,
                                                                 self,
                                                                 target)
    
    def is_elementwise_equal_to(self, target: float) -> list[bool]:
        return MiniBuffer._traverse_dims_and_compare_elementwise(0,
                                                                 0,
                                                                 MiniBuffer.CmpOp.EQ,
                                                                 self,
                                                                 target)

    # Movemenet operations

    def reshape(self, new_shape: tuple[int, ...]) -> MiniBuffer:
        return MiniBuffer(self.data, new_shape)
    
    # TODO: This was hardcoded for 3D
    def flatten(self) -> MiniBuffer:
        total_elements = math.prod(self.shape)

        return MiniBuffer(self.data, (1, 1, total_elements))

    def permute(self, order: tuple[int, ...]) -> MiniBuffer:
        new_dims = ()
        new_strides = ()

        for ord in order:
            new_dims += (self.shape[ord],)
            new_strides += (self.strides[ord],)

        return MiniBuffer(self.data, new_dims, strides=new_strides)

    # Reshape methods
    
    # NOTE: these are different from the reshape() fn. These operations
    # add/remove elements of the tensor whereas the reshape() fn just
    # changes the shape without modifying the elements.
    
    def pad(self, new_shape: tuple[int, ...]) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_reshape_op(0,
                                                                  0,
                                                                  MiniBuffer.ReshapeOp.PAD,
                                                                  new_shape,
                                                                  self)

        return MiniBuffer(out_data, new_shape)
    
    def shrink(self, new_shape: tuple[int, ...]) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_apply_reshape_op(0,
                                                                  0,
                                                                  MiniBuffer.ReshapeOp.SHRINK,
                                                                  new_shape,
                                                                  self)

        return MiniBuffer(out_data, new_shape)

    # TODO: Kinda works. Need to figure out new strides as
    # they are currently not affected by this fn.
    def expand(self, expansion_dim: int, expanded_size: int) -> MiniBuffer:
        out_shape = [dim for dim in self.shape]
        out_shape[expansion_dim] = expanded_size
        out_shape = tuple(out_shape)

        # Without this, expand wouldn't care about the strides 
        # of the new Tensor. This checks if the dimension we 
        # are expanding across is the 0th dimension.
        out_strides = [stride for stride in MiniBuffer.get_strides_from_shape(out_shape)]
        out_strides[-2], out_strides[-1] = out_strides[-1], out_strides[-2]
        out_strides = tuple(out_strides)

        out_data = self.data * expanded_size

        return MiniBuffer(out_data, out_shape, out_strides)

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
        return len(self.shape) == 1

    def is_square(self) -> bool:
        return self.shape[-2] == self.shape[-1]

    def __getitem__(self, key) -> list[float]:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        repr = str("[")

        repr += MiniBuffer._traverse_dims_and_repr(0,
                                                   0,
                                                   self)
        
        repr += "]"

        return repr

    # Helper static methods
    
    @staticmethod
    def get_strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        strides = ()
        shape_len = len(shape)

        for dim_idx in range(shape_len):
            # Stride for each dimension is calculated by taking the product 
            # of all the dimension sizes (shapes) proceeding it. The last
            # dimension always has a stride of 1.
            if dim_idx == shape_len:
                strides += (1,)
            else:
                strides += (math.prod(shape[dim_idx + 1:]),)

        return strides

    # This function iterates over the elements of all provided dimensions 
    # (taken from 'current_shape' which starts as the shape of the operand)
    # and performs the following:
    # Check if we've reached  the last dimension -> that's where the values
    # are! We iterate over the values and append them to the output list.
    # Otherwise, we iterate over the elements of the current (non-last)
    # dimension and recursively call this function with the depth_idx
    # (essentialy used for tracking recursion depth) incremented.
    # All of the calls to this function return a list of floats
    # which we can just append to the initial empty list.
    @staticmethod
    def _traverse_dims_and_apply_op(depth_idx: int,
                                    current_positions: tuple[int, ...],
                                    op: UnaryOp | BinaryOp,
                                    *operands: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(operands[0].shape) - 1:
            for val_idx in range(operands[0].shape[depth_idx]):
                x_pos = current_positions[0] + val_idx * operands[0].strides[depth_idx]
            
                if isinstance(op, MiniBuffer.UnaryOp):
                    out_data.append(MiniBuffer._apply_op_unary(op, 
                                                               operands[0].data[x_pos]))
                elif isinstance(op, MiniBuffer.BinaryOp):
                    y_pos = current_positions[1] + val_idx * operands[1].strides[depth_idx]
                    out_data.append(MiniBuffer._apply_op_binary(op, 
                                                                operands[0].data[x_pos],
                                                                operands[1].data[y_pos]))
                else:
                    assert False, f"Invalid operation: {op}."
        else:
            for dim_idx in range(operands[0].shape[depth_idx]):
                x_pos = dim_idx * operands[0].strides[depth_idx]
                current_pos = (x_pos, )

                if isinstance(op, MiniBuffer.BinaryOp):
                    y_pos = dim_idx * operands[1].strides[depth_idx]
                    current_pos += (y_pos, )

                out_data += MiniBuffer._traverse_dims_and_apply_op(depth_idx + 1,
                                                                   current_pos,
                                                                   op,
                                                                   *operands)
        
        return out_data

    # Reshape ops are different from the reshape() fn. These operations
    # add/remove elements of the tensor whereas the reshape() fn just
    # changes the shape without modifying the elements.
    @staticmethod
    def _traverse_dims_and_apply_reshape_op(depth_idx: int,
                                            current_position: int,
                                            op: ReshapeOp,
                                            new_shape: tuple[int, ...],
                                            x: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(new_shape) - 1:
            if op == MiniBuffer.ReshapeOp.PAD:
                current_dim = MiniBuffer._pad(depth_idx, 
                                              current_position, 
                                              new_shape, 
                                              x)
            elif op == MiniBuffer.ReshapeOp.SHRINK:
                current_dim = MiniBuffer._shrink(depth_idx, 
                                                 current_position, 
                                                 new_shape, 
                                                 x)
            else:
                assert False, f"Invalid operation: {op}."
            
            out_data += current_dim
        else:
            for dim_idx in range(new_shape[depth_idx]):
                current_position = dim_idx * x.strides[depth_idx]
                out_data += MiniBuffer._traverse_dims_and_apply_reshape_op(depth_idx + 1,
                                                                           current_position,
                                                                           op,
                                                                           new_shape,
                                                                           x)
        
        return out_data

    @staticmethod
    def _traverse_dims_and_compare(depth_idx: int,
                                   current_position: tuple[int, int],
                                   x: MiniBuffer,
                                   target: MiniBuffer) -> bool:
        if depth_idx == len(x.shape) - 1:
            return MiniBuffer._compare(depth_idx,
                                       current_position,
                                       x,
                                       target)
        else:
            for dim_idx in range(x.shape[depth_idx]):
                x_pos = dim_idx * x.strides[depth_idx]
                target_pos = dim_idx * target.strides[depth_idx]

                return MiniBuffer._traverse_dims_and_compare(depth_idx + 1,
                                                             (x_pos, target_pos),
                                                             x,
                                                             target)

    @staticmethod
    def _traverse_dims_and_compare_elementwise(depth_idx: int,
                                               current_position: int,
                                               op: CmpOp,
                                               x: MiniBuffer,
                                               target: float) -> bool:
        out_data = []

        if depth_idx == len(x.shape) - 1:
            for val_idx in range(x.shape[depth_idx]):
                pos = current_position + val_idx * x.strides[depth_idx]

                if op == MiniBuffer.CmpOp.LT:
                    out_data.append(x.data[pos] < target)
                elif op == MiniBuffer.CmpOp.EQ:
                    out_data.append(x.data[pos] == target)
                elif op == MiniBuffer.CmpOp.GT:
                    out_data.append(x.data[pos] > target)
                else:
                    assert False, f"Invalid operation: {op}."
        else:
            for dim_idx in range(x.shape[depth_idx]):
                current_position = dim_idx * x.strides[depth_idx]
                out_data += MiniBuffer._traverse_dims_and_compare_elementwise(depth_idx + 1,
                                                                              current_position,
                                                                              op,
                                                                              x,
                                                                              target)
        
        return out_data


    @staticmethod
    def _traverse_dims_and_masked_fill(depth_idx: int,
                                       current_position: int,
                                       mask: list[bool],
                                       value: float,
                                       x: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(x.shape) - 1:
            for val_idx in range(x.shape[depth_idx]):
                pos = current_position + val_idx * x.strides[depth_idx]
                should_fill = mask[pos]
                
                if should_fill:
                    out_data.append(value)
                else:
                    out_data.append(x.data[pos])
        else:
            for dim_idx in range(x.shape[depth_idx]):
                current_position = dim_idx * x.strides[depth_idx]
                out_data += MiniBuffer._traverse_dims_and_masked_fill(depth_idx + 1,
                                                                      current_position,
                                                                      mask,
                                                                      value,
                                                                      x)
        
        return out_data


    @staticmethod
    def _apply_op_unary(op: MiniBuffer.UnaryOp, x: float) -> float:
        if op == MiniBuffer.UnaryOp.NEG:
            return -x
        elif op == MiniBuffer.UnaryOp.LOG:
            return math.log(x, math.e)
        elif op == MiniBuffer.UnaryOp.LOG2:
            return math.log(x, 2)
        else:
            assert False, f"Reshape operation {type(op)} is not supported."

    @staticmethod
    def _apply_op_binary(op: MiniBuffer.BinaryOp, x: float, y: float) -> MiniBuffer:
        if op == MiniBuffer.BinaryOp.ADD:
            return x + y
        elif op == MiniBuffer.BinaryOp.SUB:
            return x - y
        elif op == MiniBuffer.BinaryOp.MUL:
            return x * y
        elif op == MiniBuffer.BinaryOp.DIV:
            return x / y
        elif op == MiniBuffer.BinaryOp.POW:
            return x ** y
        elif op == MiniBuffer.BinaryOp.MAX:
            return max(x, y)
        else:
            assert False, f"Reshape operation {type(op)} is not supported."

    @staticmethod
    def _pad(depth_idx: int,
             current_position: int,
             new_shape: tuple[int, ...],
             x: MiniBuffer) -> list[float]:
        current_dim = []

        for val_idx in range(new_shape[depth_idx]):
            target_position = current_position + val_idx * x.strides[depth_idx]
                
            if val_idx <= x.shape[depth_idx] and target_position < len(x.data):
                current_dim.append(x.data[target_position])
            else:
                current_dim.append(0.0)

        return current_dim

    @staticmethod
    def _shrink(depth_idx: int,
                current_position: int,
                new_shape: tuple[int, ...],
                x: MiniBuffer) -> list[float]:
        current_dim = []

        for val_idx in range(new_shape[depth_idx]):
            target_position = current_position + val_idx * x.strides[depth_idx]

            if len(current_dim) < new_shape[depth_idx]:
                current_dim.append(x.data[target_position])

        return current_dim

    @staticmethod
    def _compare(depth_idx: int,
                 current_positions: tuple[int, int],
                 x: MiniBuffer,
                 target: MiniBuffer) -> bool:
        for val_idx in range(x.shape[depth_idx]):
            x_pos = current_positions[0] + val_idx * x.strides[depth_idx]
            target_pos = current_positions[1] + val_idx * target.strides[depth_idx]

            if x.data[x_pos] != target.data[target_pos]:
                return False

        return True

    @staticmethod
    def _compare_elementwise(depth_idx: int,
                 current_position: int,
                 op: CmpOp,
                 x: MiniBuffer,
                 target: MiniBuffer) -> bool:
        for val_idx in range(x.shape[depth_idx]):
            target_position = current_position + val_idx * x.strides[depth_idx]

            if len(current_dim) < new_shape[depth_idx]:
                current_dim.append(x.data[target_position])

        return current_dim

    @staticmethod
    def _traverse_dims_and_repr(depth_idx: int,
                                current_position: int,
                                x: MiniBuffer) -> str:
        repr = ""

        if depth_idx == len(x.shape) - 1:
            for val_idx in range(x.shape[depth_idx]):
                target_position = current_position + val_idx * x.strides[depth_idx]
            
                if val_idx == (x.shape[depth_idx] - 1):
                    repr += f"{x.data[target_position]:.4f}"
                else:
                    repr += f"{x.data[target_position]:.4f}, "
        else:
            for dim_idx in range(x.shape[depth_idx]):
                # Check if we are at the beginning of the current dimension.
                # If so, add a simple opening bracket. Otherwise, add brackets
                # with spaces for proper alignment and extra newlines for tensors
                # in 3D and higher, for better visibility.
                if dim_idx == 0:
                    repr += "["
                else:
                    repr += " " * depth_idx
                    # Check if we're past 2D and if that is the case,
                    # add a extra newlines for better visibility.
                    if depth_idx + 2 < len(x.shape):
                        # Number of newlines should decrease with the
                        # depth level
                        repr += "\n" * (len(x.shape) - depth_idx - 2)
                        
                    repr += "           ["

                current_position = dim_idx * x.strides[depth_idx]
                repr += MiniBuffer._traverse_dims_and_repr(depth_idx + 1,
                                                           current_position,
                                                           x)
        
                if dim_idx == (x.shape[depth_idx] - 1):
                    repr += "]"
                else:
                    repr += "],\n"

        return repr