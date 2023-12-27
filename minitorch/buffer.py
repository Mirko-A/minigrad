from __future__ import annotations
from typing import Optional
from enum import Enum, auto
import numpy as np
import math

from minitorch.settings import DEBUG

#! WARN: Mirko, 24. 12. 2023
# In order for everything to work as expected, MiniBuffer data must remain
# contiguous at all times. That means that operations which involve permu-
# ting the MiniBuffer or manually calculating the strides must be, at the 
# end, followed by a call to MiniBuffer.contiguous() in order to create a
# new, contiguous MiniBuffer from the current one's data. 
#? NOTE: Mirko, 27. 12. 2023
# The permute() fn now ends with a call to contiguous() so it is not needed
# to do it manually after each permutation.

class MiniBuffer:
    __slots__ = ("data", "shape", "strides")
    class PadType(Enum):
        ZERO = 0
        EDGE = auto()
        # TODO: Mirko, 26. 12. 2023
        # Maybe add ramp if needed

    def __init__(self,
                 data: list[float], 
                 shape: tuple[int, ...], 
                 strides: Optional[tuple[int, ...]] = None):
        if DEBUG:
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

    #* Static MiniBuffer generation operations

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
        out_data = [value if mask[val_idx] else val for val_idx, val in enumerate(input.data)]
        
        return MiniBuffer(out_data, input.shape)

    @staticmethod
    def tril(input: MiniBuffer, diagonal: int) -> MiniBuffer:
        out_data = MiniBuffer._traverse_dims_and_tril(0,
                                                      0,
                                                      diagonal,
                                                      input)
        
        return MiniBuffer(out_data, input.shape)

    #* Unary operations

    def neg(self) -> MiniBuffer:
        out_data = [-x for x in self.data]

        return MiniBuffer(out_data, self.shape)

    def log(self) -> MiniBuffer:
        out_data = [math.log(x) for x in self.data]

        return MiniBuffer(out_data, self.shape)

    def log2(self) -> MiniBuffer:
        out_data = [math.log(x, 2) for x in self.data]

        return MiniBuffer(out_data, self.shape)

    #* Reduce operations

    def sum(self, axis: int) -> MiniBuffer:
        x = self

        # Same as input but with a 1 at the sum axis index
        out_shape = [1 if dim_idx == axis else self.shape[dim_idx] for dim_idx in range(len(self.shape))]
        dim_order = [i for i in range(len(self.shape))]

        # Permute so sum axis is last
        dim_order[axis], dim_order[-1] = dim_order[-1], dim_order[axis]
        out_shape[axis], out_shape[-1] = out_shape[-1], out_shape[axis]
        x = x.permute(dim_order)
        
        x = MiniBuffer(MiniBuffer._traverse_dims_and_sum_along_last(0,
                                                                    0,
                                                                    x), tuple(out_shape))
        
        # Permute back to original
        out_shape[axis], out_shape[-1] = out_shape[-1], out_shape[axis]

        return x.permute(dim_order)
        
    #* Binary operations

    def add(self, other: MiniBuffer) -> MiniBuffer:
        out_data = [x + y for x, y in zip(self.data, other.data)]

        return MiniBuffer(out_data, self.shape)
    
    def sub(self, other: MiniBuffer) -> MiniBuffer:
        out_data = [x - y for x, y in zip(self.data, other.data)]

        return MiniBuffer(out_data, self.shape)

    def mul(self, other: MiniBuffer) -> MiniBuffer:
        out_data = [x * y for x, y in zip(self.data, other.data)]
    
        return MiniBuffer(out_data, self.shape)
    
    def div(self, other: MiniBuffer) -> MiniBuffer:
        out_data = [x / y for x, y in zip(self.data, other.data)]

        return MiniBuffer(out_data, self.shape)
    
    def pow(self, other: MiniBuffer) -> MiniBuffer:
        out_data = [x ** y for x, y in zip(self.data, other.data)]

        return MiniBuffer(out_data, self.shape)
    
    def max(self, other: MiniBuffer) -> MiniBuffer:
        out_data = [max(x, y) for x, y in zip(self.data, other.data)]

        return MiniBuffer(out_data, self.shape)

    def is_equal_to(self, target: MiniBuffer) -> bool:
        for x in self.data:
            if x != target:
                return False
            
        return True

    def is_elementwise_greater_than(self, target: float) -> list[bool]:
        return [x > target for x in self.data]
    
    def is_elementwise_less_than(self, target: float) -> list[bool]:
        return [x < target for x in self.data]
    
    def is_elementwise_equal_to(self, target: float) -> list[bool]:
        return [x == target for x in self.data]

    #* Movemenet operations

    def reshape(self, new_shape: tuple[int, ...]) -> MiniBuffer:
        return MiniBuffer(self.data, new_shape)
    
    def flatten(self) -> MiniBuffer:
        total_elements = math.prod(self.shape)

        return MiniBuffer(self.data, (total_elements, ))

    def permute(self, order: tuple[int, ...]) -> MiniBuffer:
        out_shape = ()
        out_strides = ()

        for ord in order:
            out_shape += (self.shape[ord],)
            out_strides += (self.strides[ord],)

        result = MiniBuffer(self.data, out_shape, strides=out_strides)

        return result.contiguous(out_shape)

    #* Mutate methods
    
    #? NOTE: Mirko, 24. 12. 2023 
    # These are different from the reshape() fn. These operations
    # add/remove elements of the tensor whereas the reshape() fn just
    # changes the shape without modifying the elements.
    
    def pad(self, axis: int, pad_sizes: tuple[int, int], pad_type: MiniBuffer.PadType) -> MiniBuffer:
        x = self

        # Same as input but with a shape[axis] + sum(pad_sizes) at the sum axis index
        out_shape = [self.shape[dim_idx] + sum(pad_sizes) if dim_idx == axis else self.shape[dim_idx] for dim_idx in range(len(self.shape))]
        dim_order = [i for i in range(len(self.shape))]

        # Permute so sum axis is last
        dim_order[axis], dim_order[-1] = dim_order[-1], dim_order[axis]
        out_shape[axis], out_shape[-1] = out_shape[-1], out_shape[axis]
        x = x.permute(dim_order)

        x = MiniBuffer(MiniBuffer._traverse_dims_and_pad_along_last(0,
                                                                    0,
                                                                    pad_sizes,
                                                                    pad_type,
                                                                    x), tuple(out_shape))
        
        # Permute back to original

        return x.permute(dim_order)
    
    def shrink(self, axis: int, shrink_sizes: [int, int]) -> MiniBuffer:
        x = self

        # Same as input but with a 1 at the sum axis index
        out_shape = [self.shape[dim_idx] - sum(shrink_sizes) if dim_idx == axis else self.shape[dim_idx] for dim_idx in range(len(self.shape))]
        dim_order = [i for i in range(len(self.shape))]

        # Permute so sum axis is last
        dim_order[axis], dim_order[-1] = dim_order[-1], dim_order[axis]
        out_shape[axis], out_shape[-1] = out_shape[-1], out_shape[axis]
        x = x.permute(dim_order)
        
        x = MiniBuffer(MiniBuffer._traverse_dims_and_shrink_along_last(0,
                                                                       0,
                                                                       shrink_sizes,
                                                                       x), tuple(out_shape))
        
        # Permute back to original

        return x.permute(dim_order)

    def expand(self, axis: int, expanded_size: int) -> MiniBuffer:
        out_data = self.data * expanded_size

        out_shape = [dim for dim in self.shape]
        out_shape[axis] = expanded_size

        #? NOTE: Mirko, 24. 12. 2023 
        # Since we're just multiplying the data array by
        # expanded_size, somehow we need to preserve the
        # meaning of the original shape and strides. Other-
        # wise, expanding a 1x3 and a 3x1 tensor would res-
        # ult in the same output, which is wrong.
        # The correct strides for the output are same as
        # the input strides, with the stride at position
        # pos=expansion_axis being the product of all dims
        # of the original shape except the one we're expan-
        # ding along. This makes sense because we expand by
        # simply duplicating the original data expanded_size
        # times.
        out_strides = [stride for stride in self.strides]
        corrected_input_shape = [1 if i == axis else self.shape[i] for i in range(len(self.shape))]
        out_strides[axis] = math.prod(corrected_input_shape)

        result = MiniBuffer(out_data, tuple(out_shape), tuple(out_strides))

        return result.contiguous(tuple(out_shape))

    #* Unary operator magic methods

    def __neg__(self):
        return self.neg()

    #* Binary operator magic methods

    def __add__(self, other):
        if DEBUG:
            assert isinstance(other, MiniBuffer), f"Cannot perform addition with MiniBuffer and {type(other)}."

        return self.add(other)
    
    def __sub__(self, other):
        if DEBUG:
            assert isinstance(other, MiniBuffer), f"Cannot perform subtraction with MiniBuffer and {type(other)}."

        return self.sub(other)

    def __mul__(self, other):
        if DEBUG:
            assert isinstance(other, MiniBuffer), f"Cannot perform multiplication with MiniBuffer and {type(other)}."

        return self.mul(other)

    def __truediv__(self, other):
        if DEBUG:
            assert isinstance(other, MiniBuffer), f"Cannot perform division with MiniBuffer and {type(other)}."

        return self.div(other)
    
    def __pow__(self, other):
        if DEBUG:
            assert isinstance(other, MiniBuffer), f"Cannot perform exponentiation with MiniBuffer and {type(other)}."

        return self.pow(other)

    def __lt__(self, other):
        if DEBUG:
            assert isinstance(other, (int, float)), f"Invalid type for Tesnor less-than: {type(other)}. Expected int or float."
        if isinstance(other, int):
            other = float(other)

        return self.is_elementwise_less_than(other)
    
    def __gt__(self, other):
        if DEBUG:
            assert isinstance(other, (int, float)), f"Invalid type for Tesnor greater-than: {type(other)}. Expected int or float."
        if isinstance(other, int):
            other = float(other)

        return self.is_elementwise_greater_than(other)

    #* Utility

    #? NOTE: Mirko, 24. 12. 2023
    # The purpose of this function is to create a contiguous MiniBuffer
    # from one that is not contiguous anymore. This can happen as a result
    # of any operation* that involves manually calculating the new strides.
    # As most operations on MiniBuffers rely on them being contiguous, all
    # such* operations shall be followed up with a call to this function.
    def contiguous(self, shape: tuple[int, ...]) -> MiniBuffer:
        return MiniBuffer(MiniBuffer._traverse_dims_and_collect_data(0, 0, self), shape)

    def is_scalar(self) -> bool:
        return len(self.shape) == 1

    def is_square(self) -> bool:
        assert len(self.shape) >= 2, f"Cannot check for squareness on a {len(self.shape)}D Tensor. Expected 2D or higher."
        return self.shape[-2] == self.shape[-1]

    def __getitem__(self, keys: tuple[int, ...]) -> list[float]:
        item_pos = 0

        for dim_idx, key in enumerate(keys):
            item_pos += self.strides[dim_idx] * key
            
        return self.data[item_pos]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        repr = str("[")

        repr += MiniBuffer._traverse_dims_and_repr(0,
                                                   0,
                                                   self)
        
        repr += "]"

        return repr

    #* Helper static methods
    
    @staticmethod
    def get_strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        strides = ()
        shape_len = len(shape)

        for dim_idx in range(shape_len):
            # Stride for each dimension is calculated by taking the product 
            # of all the dimension sizes (shapes) proceeding it. The last
            # dimension always has a stride of 1.
            if dim_idx == (shape_len - 1):
                strides += (1,)
            else:
                strides += (math.prod(shape[dim_idx + 1:]),)

        return strides

    @staticmethod
    def _traverse_dims_and_collect_data(depth_idx: int,
                                        current_position: int,
                                        x: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(x.shape) - 1:
            for val_idx in range(x.shape[depth_idx]):
                val_pos = current_position + val_idx * x.strides[depth_idx]
            
                out_data.append(x.data[val_pos])
        else:
            for dim_idx in range(x.shape[depth_idx]):
                next_pos = current_position + dim_idx * x.strides[depth_idx]

                out_data += MiniBuffer._traverse_dims_and_collect_data(depth_idx + 1,
                                                                       next_pos,
                                                                       x)
        
        return out_data

    @staticmethod
    def _traverse_dims_and_sum_along_last(depth_idx: int,
                                          current_position: int,
                                          x: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(x.shape) - 1:
            sum = 0.0

            for val_idx in range(x.shape[depth_idx]):
                val_pos = current_position + val_idx * x.strides[depth_idx]
                sum += x.data[val_pos]

            out_data.append(sum)
        else:
            for dim_idx in range(x.shape[depth_idx]):
                next_pos = current_position +  dim_idx * x.strides[depth_idx]

                out_data += MiniBuffer._traverse_dims_and_sum_along_last(depth_idx + 1,
                                                                         next_pos,
                                                                         x)

        return out_data

    @staticmethod
    def _traverse_dims_and_pad_along_last(depth_idx: int,
                                          current_position: int,
                                          pad_sizes: tuple[int, int],
                                          pad_type: PadType,
                                          x: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(x.shape) - 1:
            out_row = []

            for val_idx in range(x.shape[depth_idx]):
                val_pos = current_position + val_idx * x.strides[depth_idx]
                out_row.append(x.data[val_pos])

            if pad_type == MiniBuffer.PadType.ZERO:
                out_row = [0]*pad_sizes[0] + out_row + [0]*pad_sizes[1]
            elif pad_type == MiniBuffer.PadType.EDGE:
                out_row = out_row[0]*pad_sizes[0] + out_row + out_row[-1]*pad_sizes[1]
            else:
                assert False, f"Invalid pad type: {pad_type}."
            
            out_data += out_row
        else:
            for dim_idx in range(x.shape[depth_idx]):
                next_pos = current_position + dim_idx * x.strides[depth_idx]
                out_data += MiniBuffer._traverse_dims_and_pad_along_last(depth_idx + 1,
                                                                         next_pos,
                                                                         pad_sizes,
                                                                         pad_type,
                                                                         x)
        
        return out_data

    @staticmethod
    def _traverse_dims_and_shrink_along_last(depth_idx: int,
                                             current_position: int,
                                             shrink_sizes: tuple[int, int],
                                             x: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(x.shape) - 1:
            out_row = []

            for val_idx in range(x.shape[depth_idx]):
                val_pos = current_position + val_idx * x.strides[depth_idx]
                out_row.append(x.data[val_pos])
            
            out_data += out_row[shrink_sizes[0]:len(out_row) - shrink_sizes[1]]
        else:
            for dim_idx in range(x.shape[depth_idx]):
                next_pos = current_position + dim_idx * x.strides[depth_idx]
                out_data += MiniBuffer._traverse_dims_and_shrink_along_last(depth_idx + 1,
                                                                            next_pos,
                                                                            shrink_sizes,
                                                                            x)
        
        return out_data

    @staticmethod
    def _traverse_dims_and_tril(depth_idx: int,
                                current_position: int,
                                diagonal: str,
                                x: MiniBuffer) -> list[float]:
        out_data = []

        if depth_idx == len(x.shape) - 2:
                out_data += MiniBuffer._tril(depth_idx,
                                             current_position,
                                             diagonal,
                                             x)
        else:
            for dim_idx in range(x.shape[depth_idx]):
                next_pos = current_position + dim_idx * x.strides[depth_idx]
                out_data += MiniBuffer._traverse_dims_and_tril(depth_idx + 1,
                                                               next_pos,
                                                               diagonal,
                                                               x)
        
        return out_data
    
    #? NOTE: Mirko, 24. 12. 2023 
    # This only works with contiguous MiniBuffers, so
    # you better make sure to keep them contiguous at
    # all times.
    def _tril(depth_idx: int,
              current_position: int,
              diagonal: int,
              x: MiniBuffer) -> list[float]:
        out_data = []
        tril_cursor = 1 + diagonal

        for row_idx in range(x.shape[depth_idx]):
            out_row = []
            row_pos = current_position + row_idx * x.strides[depth_idx]

            for val_idx in range(x.shape[depth_idx + 1]):
                val_pos = row_pos + val_idx * x.strides[depth_idx + 1]
                should_keep_value = val_idx < tril_cursor
                out_row.append(x.data[val_pos] if should_keep_value else 0.0)
            
            out_data += out_row
            tril_cursor += 1
            
        return out_data

    @staticmethod
    def _traverse_dims_and_repr(depth_idx: int,
                                current_position: int,
                                x: MiniBuffer) -> str:
        repr = ""

        if depth_idx == len(x.shape) - 1:
            for val_idx in range(x.shape[depth_idx]):
                val_pos = current_position + val_idx * x.strides[depth_idx]
            
                if val_idx == (x.shape[depth_idx] - 1):
                    repr += f"{x.data[val_pos]:.4f}"
                else:
                    repr += f"{x.data[val_pos]:.4f}, "
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

                x_pos = current_position + dim_idx * x.strides[depth_idx]
                repr += MiniBuffer._traverse_dims_and_repr(depth_idx + 1,
                                                           x_pos,
                                                           x)
        
                if dim_idx == (x.shape[depth_idx] - 1):
                    repr += "]"
                else:
                    repr += "],\n"

        return repr