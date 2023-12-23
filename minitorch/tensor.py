from __future__ import annotations
from typing import Optional, Type
from random import gauss, uniform
import math

from minitorch.buffer import MiniBuffer
from minitorch import helpers

class Function:
    def __init__(self, *inputs: Tensor):
        self.inputs_need_grad = [input.requires_grad for input in inputs]
        self.output_requires_grad = True if any(self.inputs_need_grad) else False
        if self.output_requires_grad:
            self.parents = inputs

    def forward(self, *args, **kwargs) -> Tensor: assert False, f"forward not implemented for {type(self)}"
    def backward(self, *args, **kwargs) -> Tensor: assert False, f"backward not implemented for {type(self)}"

    @classmethod
    def apply(fn_ctor: Type[Function], *inputs: Tensor, **kwargs) -> Tensor: 
        # fn_ctor is the constructor of the child class from which 'apply' is called
        ctx = fn_ctor(*inputs)
        result = Tensor(ctx.forward(*[input.data for input in inputs], **kwargs), ctx.output_requires_grad)
        
        # Keep the reference to the function which created the result
        # Used for autograd
        if ctx.output_requires_grad:
            result._ctx = ctx
        
        return result

import minitorch.ops as ops

class Tensor:
    def __init__(self, data: float | int | list | MiniBuffer, requires_grad: bool = False):
        if isinstance(data, MiniBuffer):
            self.data = data
        elif isinstance(data, float):
            self.data = MiniBuffer(data, (1,))
        elif isinstance(data, int):
            self.data = MiniBuffer([float(data)], (1,))
        elif isinstance(data, list):
            self.data = MiniBuffer.np_load(data)
        else:
            assert False, "Cannot construct Tensor3D with given data. Expected: int | float | list(nested) | MiniBuffer."
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        # Internal variable used for autograd graph construction
        self._ctx: Optional[Function] = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
    
    @property
    def T(self) -> Tensor:
        return self.transpose()

    # Static Matrix generation methods
    # TODO: Implement: one_hot

    @staticmethod
    def fill(shape: tuple[int, ...], value: float, requires_grad: bool = False) -> Tensor:
        return Tensor(MiniBuffer.fill(shape, value), requires_grad)

    @staticmethod
    def zeros(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
        return Tensor.fill(shape, 0.0, requires_grad)

    @staticmethod
    def ones(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
        return Tensor.fill(shape, 1.0, requires_grad)

    @staticmethod
    def arange(start: int, end: int, requires_grad: bool = False):
        return Tensor([i for i in range(start, end)], requires_grad)

    @staticmethod
    def randn(shape: tuple[int, ...], mean: float = 0.0, std_dev: float = 1.0, requires_grad: bool = False) -> Tensor:
        data = MiniBuffer([gauss(mean, std_dev) for _ in range(math.prod(shape))], shape)
        return Tensor(data, requires_grad)

    @staticmethod
    def uniform(shape: tuple[int, ...], low: float, high: float, requires_grad: bool = False) -> Tensor:
        data = data = MiniBuffer([uniform(low, high) for _ in range(math.prod(shape))], shape)
        return Tensor(data, requires_grad)

    @staticmethod
    def masked_fill(input: Tensor, mask: list[bool], value: float) -> Tensor:
        assert len(mask) == len(input.data), \
               f"Cannot mask {input.shape} Tensor with mask of length {len(mask)}"
        assert all(isinstance(mask_val, bool) for mask_val in mask), \
               f"Invalid mask type provided. Expected list[bool]"
        return Tensor(MiniBuffer.masked_fill(input.data, mask, value), input.requires_grad)

    @staticmethod
    def replace(input: Tensor, target: float, new: float) -> Tensor:
        return Tensor(MiniBuffer.replace(input, target, new), input.requires_grad)
    
    # TODO: Not implemented!!
    @staticmethod
    def tril(input: Tensor) -> Tensor:
        assert input.is_square(), "Cannot apply tril to non-square matrices."
      
        def tril_main_diagonal(input: Tensor) -> Tensor:
            out_data = []
            tril_cursor = 1

            for row in input.data:
                out_row = []

                for value_pos, value in enumerate(row):
                    should_keep_value = value_pos < tril_cursor
                    out_row.append(value if should_keep_value else 0.0)

                out_data.append(out_row)
                tril_cursor += 1
            
            return Tensor(out_data, input.requires_grad)
      
        def tril_anti_diagonal(input: Tensor) -> Tensor:
            out_data = []
            tril_cursor = 0

            for row in input.data:
                out_row = []

                for value_pos, value in enumerate(row):
                    should_replace_value = value_pos < tril_cursor
                    out_row.append(0.0 if should_replace_value else value)

                out_data.append(out_row)
                tril_cursor += 1
          
            return Tensor(out_data, input.requires_grad)

        output = tril_main_diagonal(input) if diagonal == Tensor.Diagonal.MAIN else tril_anti_diagonal(input)

        return output

    # Movement methods
    
    # NOTE: The way this version of reshape (with *args) works
    # isn't super clear, it was simply stolen from tinygrad. 
    # The original had just the new shape and it applied the
    # Reshape operation on it. The current version was taken
    # from tinygrad because it is needed for matmul.
    def reshape(self, new_shape: tuple[int, ...], *args) -> Tensor:
        new_shape = helpers.argfix(new_shape, *args)
        assert 0 not in new_shape, f"zeros not allowed in shape {new_shape}"
        
        return ops.Reshape.apply(self, new_shape=tuple([-math.prod(self.shape) // math.prod(new_shape) if s == -1 else s for s in new_shape]))
    
    def flatten(self) -> Tensor:
        total_elements = math.prod(self.shape)

        return ops.Reshape.apply(self, new_shape=(total_elements, ))

    def permute(self, order: tuple[int, ...]) -> Tensor:
        assert len(order) >= len(self.shape), \
                f"Cannot permute Tensor. new shape dimensionality {len(order)} is smaller than original one {len(self.shape)}"
        x = self
        shape_diff = len(order) - len(x.shape)
        
        if shape_diff > 0:
            x = Tensor.pad_shapes(shape_diff, x)

        return ops.Permute.apply(x, order=order)

    def transpose(self, axis0 = -2, axis1 = -1) -> Tensor:
        x = self
        shape_len = len(x.shape)

        if shape_len < 2:
            x = Tensor.pad_shapes(2 - shape_len, x)

        order = [i for i in range(len(x.shape))]
        order[axis0], order[axis1] = order[axis1], order[axis0]
        return ops.Permute.apply(x, order=order)

    # Reshape methods

    # NOTE: these are different from the reshape() fn. These operations
    # add/remove elements of the tensor whereas the reshape() fn just
    # changes the shape without modifying the elements.
    
    def pad(self, new_shape: tuple[int, ...]) -> Tensor:
        assert len(new_shape) >= len(self.shape), \
            f"Cannot pad, new shape dimensionality {new_shape} is smaller than original one {self.shape}."
        assert isinstance(new_shape, tuple) and all(isinstance(dim, int) for dim in new_shape), \
                f"Cannot pad, new shape expected type is tuple[int, ...] but got type{new_shape}."
        x = self
        shape_diff = len(new_shape) - len(x.shape)
        
        if shape_diff > 0:
            x = Tensor.pad_shapes(shape_diff, x)

        return ops.Pad.apply(x, new_shape=new_shape)

    def shrink(self, new_shape: tuple[int, ...]) -> Tensor:
        assert len(new_shape) <= len(self.shape), \
            f"Cannot shrink, new shape dimensionality {new_shape} is greater than original one {self.shape}."
        assert isinstance(new_shape, tuple) and all(isinstance(dim, int) for dim in new_shape), \
                f"Cannot shrink, new shape expected type is tuple[int, ...] but got type{new_shape}."
        x = self
        shape_diff = len(new_shape) - len(x.shape)
        
        if shape_diff > 0:
            x = Tensor.squeeze_shapes(shape_diff, x)

        return ops.Shrink.apply(x, new_shape=new_shape)
    
    def expand(self, new_shape: tuple[int, ...]) -> Tensor:
        assert len(new_shape) >= len(self.shape), \
            f"Cannot pad, new shape dimensionality {new_shape} is smaller than original one {self.shape}."
        assert isinstance(new_shape, tuple) and all(isinstance(dim, int) for dim in new_shape), \
                f"Cannot expand, new shape expected type is tuple[int, ...] but got type{new_shape}."
        x = self
        shape_diff = len(new_shape) - len(x.shape)
        
        if shape_diff > 0:
            x = Tensor.pad_shapes(shape_diff, x)

        for dim_idx, (new_dim, current_dim) in enumerate(zip(new_shape, x.shape)):
            if new_dim > current_dim:
                assert current_dim == 1, "Cannot expand along a non-singular dimension."
                x = ops.Expand.apply(x, expansion_axis=dim_idx, expanded_size=new_dim)
        
        return x
    
    # Unary operations

    def neg(self) -> Tensor:
        return ops.Neg.apply(self)

    def log(self) -> Tensor:
        return ops.Log.apply(self)

    def log2(self) -> Tensor:
        return ops.Log2.apply(self)
    
    # Reduce operations

    def sum(self, sum_axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        if sum_axis is not None:
            assert isinstance(sum_axis, int), f"Cannot sum Tensor, invalid axis provided. Expected int but got {type(sum_axis)}."
            assert sum_axis < len(self.shape), f"Cannot sum Tensor, invalid axis provided. Tensor shape is {self.shape} but {sum_axis}th dimension was provided."
        
            if sum_axis < 0:
                sum_axis = len(self.shape) + sum_axis

        if sum_axis is None:
            shape_squeezed = (1,)
        else:
            shape_squeezed = [s for i,s in enumerate(self.shape) if i != sum_axis]
        
        sum = ops.Sum.apply(self, sum_axis=sum_axis)
        return sum if keepdims else ops.Reshape.apply(sum, new_shape=tuple(shape_squeezed))

    # Binary operations

    def add(self, other: Tensor, reverse: bool = False) -> Tensor:
        x, y = self, other

        if reverse:
            x, y = y, x

        return ops.Add.apply(*x._broadcasted(y))
    
    def sub(self, other: Tensor, reverse: bool = False) -> Tensor:
        x, y = self, other

        if reverse:
            x, y = y, x

        return ops.Sub.apply(*x._broadcasted(y))

    def mul(self, other: Tensor, reverse: bool = False) -> Tensor:
        x, y = self, other

        if reverse:
            x, y = y, x

        return ops.Mul.apply(*x._broadcasted(y))
    
    def div(self, other: Tensor, reverse: bool = False) -> Tensor:
        x, y = self, other

        if reverse:
            x, y = y, x

        return ops.Div.apply(*x._broadcasted(y))

    def pow(self, other: Tensor, reverse: bool = False) -> Tensor:
        x, y = self, other

        if reverse:
            x, y = y, x

        return ops.Pow.apply(*x._broadcasted(y), is_exp=reverse)

    def exp(self) -> Tensor:
        return math.e ** self

    # TODO: Not implemented!!
    def matmul(self, other: Tensor) -> Tensor:
        x, y = self, other
        n1, n2 = len(x.shape), len(y.shape)
        
        assert n1 > 1 and n2 > 1, \
            f"both arguments to matmul need to be at least 2D, but they are {n1}D and {n2}D"
        assert x.shape[-1] == y.shape[-2], \
            f"Input Tensor shapes {x.shape} and {y.shape} cannot be multiplied ({x.shape[-1]} != {y.shape[-2]})"
        
        # NOTE: This last part isn't super clear, it was simply
        # stolen from tinygrad. But my guess is that it is an
        # n-dimensional analogy to the way you can perfom matrix
        # multiplication between A[m, n] and B[n, p] by performing
        # the following operations:
        # 1) A = reshape(A, (1, m, n))
        # 2) B = reshape(B, (n, p, 1))
        # 3) B = transpose(B)
        # 4) C = (A*B).sum(-1)
        x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
        y = y.reshape(*y.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *y.shape[-2:]).transpose()

        return (x*y).sum(-1)


    # Activation functions
    # TODO: Implement tanh, softmax
    
    def sigmoid(self) -> Tensor:
        return ops.Sigmoid.apply(self)

    def relu(self) -> Tensor:
        return ops.Relu.apply(self)
    
    # Cost functions
    # TODO: Implement MSE, cross_entropy

    # Backpropagation

    def toposort(self):
        def _toposort(node: Tensor, visited: set[Tensor], nodes: list[Tensor]):
            visited.add(node)

            if getattr(node, "_ctx", None):
                for parent in node._ctx.parents:
                    if parent not in visited: 
                        _toposort(parent, visited, nodes)

                nodes.append(node)

            return nodes
      
        return _toposort(self, set(), [])

    def backward(self):
        assert self.is_scalar(), f"Backward can only be called for scalar tensors, but it has shape {self.shape})"

        self.grad = Tensor(1, requires_grad=False)
        autograd_graph = self.toposort()

        for node in reversed(autograd_graph):
            assert node.grad is not None

            grads = node._ctx.backward(node.grad.data)
            grads = [Tensor(g, requires_grad=False) if g is not None else None
                        for g in ([grads] if len(node._ctx.parents) == 1 else grads)]
            
            for parent, grad in zip(node._ctx.parents, grads):
                if grad is not None and parent.requires_grad:
                    assert grad.shape == parent.shape, f"Grad shape must match Matrix shape, {grad.shape} != {parent.shape}"
                    parent.grad = grad if parent.grad is None else (parent.grad + grad)
            
            # TODO: Bring back if needed (needs __deletable__ = '_ctx'?)
            # del node._ctx

    # Broadcasting

    def _broadcasted(self, y: Tensor) -> tuple[Tensor, Tensor]:
        x: Tensor = self
    
        if (xshape:=x.shape) == (yshape:=y.shape):
            return (x, y)
        
        shape_delta = len(xshape) - len(yshape)

        if shape_delta > 0:
            new_shape = (1,) * shape_delta + yshape 
            y = y.reshape(new_shape)
        elif shape_delta < 0:
            new_shape = (1,) * -shape_delta + xshape 
            x = x.reshape(new_shape)

        if (xshape:=x.shape) == (yshape:=y.shape): 
            return (x, y)
    
        shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])

        if xshape != shape_ret: 
            x = x.expand(shape_ret)
        if yshape != shape_ret: 
            y = y.expand(shape_ret)

        #print(x)
        #print(y)
        return (x, y)

    # Unary operator magic methods

    def __neg__(self) -> Tensor:
        return self.neg()

    # TODO: Not implemented!! Could use at(x, y, z)
    def __getitem__(self, key) -> list[float]:
        return self.data[key]
    
    # Binary operator magic methods

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return self.add(other)
    
    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return self.add(other, True)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return self.sub(other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return self.sub(other, True)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, False)

        return self.mul(other)
    
    def __rmul__(self, other):
        other = Tensor(other, False)

        return self.mul(other, True)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, False)

        return self.div(other)
    
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, False)

        return self.div(other, True)
    
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, False)

        return self.pow(other)

    def __rpow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, False)

        return self.pow(other, True)

    def __matmul__(self, other):
        assert isinstance(other, Tensor), f"Cannot perform Tensor multiplication with type {type(other)}"

        return self.matmul(other)
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            assert self.shape == other.shape, f"Cannot compare {self.shape} and {other.shape} Tensors. Shapes must match."
            return self.data.is_equal_to(other.data)
        elif isinstance(other, (int, float)):
            if isinstance(other, int):
                other = float(other)
            return self.data.is_elementwise_equal_to(other)
        else:
            assert False, f"Invalid type for Tensor equality: {type(other)}. Expected Tensor, int or float."

    def __lt__(self, other):
        assert isinstance(other, (int, float)), f"Invalid type for Tesnor less-than: {type(other)}. Expected int or float."
        if isinstance(other, int):
            other = float(other)

        return self.data.is_elementwise_less_than(other)
    
    def __gt__(self, other):
        assert isinstance(other, (int, float)), f"Invalid type for Tesnor greater-than: {type(other)}. Expected int or float."
        if isinstance(other, int):
            other = float(other)

        return self.data.is_elementwise_greater_than(other)

    def __hash__(self):
        return id(self)

    # Utility

    @staticmethod
    def pad_shapes(shape_diff: int, x: Tensor) -> Tensor:
        padded_shape = (1,) * shape_diff + x.shape
        return ops.Reshape.apply(x, new_shape=padded_shape)

    @staticmethod
    def squeeze_shapes(shape_diff: int, x: Tensor) -> Tensor:
        squeezed_shape = x.shape[shape_diff:]
        return ops.Reshape.apply(x, new_shape=squeezed_shape)

    def is_scalar(self) -> bool:
        return self.data.is_scalar()
    
    def is_square(self) -> bool:
        return self.data.is_square()

    def __repr__(self) -> str:
        return f"<Tensor:  {self.data} with grad {self.grad.data if self.grad else None}>"

#     @staticmethod
#     def cat(matrices: list[Matrix], dim: int = 0) -> Matrix:
#         assert dim in Matrix._VALID_DIM_VALUES, "Invalid dimension value provided. Expected: 0 or 1."
#         assert all(isinstance(m, Matrix) for m in matrices), f"Cannot concatenate Matrix with other data types."

#         def cat_rows(matrices: list[Matrix]) -> Matrix:
#             rows = matrices[0].shape.row
#             requires_grad = matrices[0].requires_grad
#             assert all(m.shape.row == rows and m.requires_grad == requires_grad for m in matrices)

#             out_data = []

#             for row in range(rows):
#                 out_row = sum((m[row] for m in matrices), [])
#                 out_data.append(out_row)

#             return Matrix(out_data, requires_grad)

#         def cat_cols(matrices: list[Matrix]) -> Matrix:
#             cols = matrices[0].shape.col 
#             requires_grad = matrices[0].requires_grad
#             assert all(m.shape.row == cols and m.requires_grad == requires_grad for m in matrices)

#             out_data = [row for m in matrices for row in m.data]
#             return Matrix(out_data, requires_grad)

#         if dim == 0:
#             return cat_cols(matrices)
#         else:
#             return cat_rows(matrices)

#     def softmax(self, dim: int = 0):
#         assert dim in Matrix._VALID_DIM_VALUES, "Invalid dimension value provided. Expected: 0 or 1."

#         input = self if dim == 1 else self.T()
#         input_exp = input.exp()
#         input_exp_sums = input_exp.sum(dim=1).item()
#         out_data = []
        
#         for row_exp, row_exp_sum in zip(input_exp.data, input_exp_sums):
#             out_row = []

#             for value_exp in row_exp:
#                 probability = value_exp / row_exp_sum
#                 out_row.append(probability)

#             out_data.append(out_row)

#         out_mat = Matrix(out_data, self.requires_grad)

#         return out_mat if dim == 1 else out_mat.T()

#     # Cost functions
    
#     def cross_entropy(self, target: Matrix):
#         assert isinstance(target, Matrix), f"Cannot perform Cross-Entropy on target type {type(target)}"
#         assert self._dims_match_with(target), "Cannot perform Cross-Entropy. Dimensions of input don't match with target."
#         # NOTE: PyTorch uses base e here, might be relevant later
#         input_log = self.log(2)
#         out_data = []
        
#         for target_row, input_log_row in zip(target.data, input_log.data):
#             cross_entropy_sum = 0
            
#             for target_value, input_log_value in zip(target_row, input_log_row):
#                 cross_entropy = target_value * input_log_value
#                 cross_entropy_sum += cross_entropy
                
#             out_data.append(-cross_entropy_sum)
            
#         return Matrix([out_data], self.requires_grad)
    
#     def MSE(self, target: Matrix):
#         assert isinstance(target, Matrix), f"Cannot perform MSE on target type {type(target)}"
#         assert self._dims_match_with(target), "Cannot perform MSE. Dimensions of input don't match with target."
        
#         MSE = []
        
#         for input_row, target_row in zip(self.data, target.data):
#             row_error_sum = 0
            
#             for input_value, target_value in zip(input_row, target_row):
#                 squared_error = (target_value - input_value) ** 2
#                 row_error_sum += squared_error
                
#             MSE.append(row_error_sum / self.shape.col)
        
#         return Matrix([MSE], self.requires_grad)
