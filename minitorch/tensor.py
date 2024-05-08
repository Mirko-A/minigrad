from __future__ import annotations
from collections import defaultdict, deque
from functools import reduce
from itertools import accumulate
from typing import Optional, Type
from random import gauss, uniform
import numpy as np
import math

from minitorch.storage import Storage
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
        result = Tensor(ctx.forward(*[input._storage for input in inputs], **kwargs), requires_grad=ctx.output_requires_grad)
        
        # Keep the reference to the function which created the result
        # Used for autograd
        if ctx.output_requires_grad:
            result._ctx = ctx
        
        return result

import minitorch.ops as ops

class Tensor:
    Value = Storage.Value
    ArrayLike = Storage.ArrayLike
    Dtype = Storage.Dtype
    
    __slots__ = ("_storage", "requires_grad", "grad", "_ctx")
    __deletable__ = ("_ctx",)

    def __init__(self, data: Value | ArrayLike | Storage, dtype: Optional[Dtype] = None, requires_grad: bool = False):
        if isinstance(data, Storage):
            self._storage = data
        else:
            self._storage = Storage(data, dtype)
            
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        # Internal variable used for autograd graph construction
        self._ctx: Optional[Function] = None

    @property
    def _np(self):
        return self._storage._np

    @property
    def shape(self):
        return self._storage.shape

    @property
    def size(self):
        return self._storage.size

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def T(self) -> Tensor:
        return self.transpose()

    #* Static methods

    @staticmethod
    def full(shape: tuple[int, ...], value: Value, dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        return Tensor(Storage.full(shape, value, dtype), requires_grad=requires_grad)

    @staticmethod
    def full_like(other: Tensor, value: Value, dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        return Tensor(Storage.full_like(other, value, dtype), requires_grad=requires_grad)

    @staticmethod
    def zeros(shape: tuple[int, ...], dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        return Tensor.full(shape, 0, dtype, requires_grad)

    @staticmethod
    def ones(shape: tuple[int, ...], dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        return Tensor.full(shape, 1, dtype, requires_grad)

    @staticmethod
    def arange(start: int, end: Optional[int] = None, dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        if end is None:
            start, end = 0, start
        return Tensor(Storage.arange(start, end, dtype), requires_grad=requires_grad)

    @staticmethod
    def one_hot(n_classes: int, hot_class: int, dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        data = [1 if i == hot_class else 0 for i in range(n_classes)]
        return Tensor(data, dtype, requires_grad) 

    @staticmethod
    def randn(shape: tuple[int, ...], mean: float = 0.0, std_dev: float = 1.0, dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        data = np.reshape(np.array([gauss(mean, std_dev) for _ in range(math.prod(shape))]), shape)
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)

    @staticmethod
    def uniform(shape: tuple[int, ...], low: float, high: float, dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor:
        data = np.reshape(np.array([uniform(low, high) for _ in range(math.prod(shape))]), shape)
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)

    def masked_fill(self, mask: Tensor, value: Value) -> Tensor:
        old, new = self, Tensor(value)
        return mask.where(new, old)

    def replace(self, target: Value, new: Value) -> Tensor:
        old, target, new = self, Tensor(target), Tensor(new)
        return (old == target).where(new, old)

    @staticmethod
    def _tri(row: int, col: int, offset: int=0, dtype: Optional[Dtype] = None, requires_grad: bool = False) -> Tensor: 
        tri_mask = (Tensor.arange(row, dtype=dtype, requires_grad=requires_grad).unsqueeze(1).expand((row,col)) > Tensor.arange(-offset, col-offset, dtype=dtype, requires_grad=requires_grad).unsqueeze(0).expand((row,col)))
        return tri_mask.where(Tensor(1), Tensor(0))
    
    def tril(self, k: int=0) -> Tensor:
        assert helpers.all_int(self.shape), f"does not support symbolic shape {self.shape}"

        return Tensor._tri(self.shape[-2], self.shape[-1], offset=k+1, dtype=self.dtype).where(self, Tensor.full_like(self, 0.0))

    def triu(self, k: int=0) -> Tensor:
        assert helpers.all_int(self.shape), f"does not support symbolic shape {self.shape}"

        return Tensor._tri(self.shape[-2], self.shape[-1], offset=k, dtype=self.dtype).where(Tensor.full_like(self, 0.0), self)

    def cat(self, others: list[Tensor], axis: int = 0) -> Tensor:
        axis = (axis + len(self.shape)) if axis < 0 else axis
        assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != axis) for y in others)
        all_tensors: list[Tensor] = [self, *others]
        assert all(t.shape for t in all_tensors), "ERROR: Zero-dimensional Tensor cannot be concatenated"

        shapes = [s.shape[axis] for s in all_tensors]
        shape_cumsum = [0, *accumulate(shapes)]
        slc = [[(0, 0) for _ in self.shape] for _ in all_tensors]
        
        for shp,k,s in zip(shapes, shape_cumsum[:-1], slc):
            s[axis] = (k, shape_cumsum[-1] - k - shp)
        
        return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(all_tensors, slc)])

    @staticmethod
    def concat(tensors: list[Tensor], axis: int = 0) -> Tensor:
        first = tensors[0]
        rest = [tensor for tensor in tensors[1:]]

        return first.cat(rest, axis)

    @staticmethod
    def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
        first = tensors[0].unsqueeze(axis)
        rest = [tensor.unsqueeze(axis) for tensor in tensors[1:]]

        return first.cat(rest, axis)

    #* Movement methods
    
    def permute(self, order: tuple[int, ...]) -> Tensor:
        return ops.Permute.apply(self, order=order)

    def transpose(self, axis0 = -2, axis1 = -1) -> Tensor:
        order = [i for i in range(len(self.shape))]
        order[axis0], order[axis1] = order[axis1], order[axis0]

        return ops.Permute.apply(self, order=tuple(order))

    def flip(self, axis: tuple[int, ...], *args) -> Tensor:
        return ops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in helpers.argfix(axis, *args)])

    #* Mutate methods
    
    #? NOTE: Mirko, 1. 1. 2024.
    # This function accepts a new shape as a tuple[int] but can
    # optionally accept any number of additional integers and
    # those will be appended to the new shape.
    def reshape(self, new_shape: tuple[int, ...], *args: int) -> Tensor:
        return ops.Reshape.apply(self, new_shape=tuple(helpers.argfix(new_shape, *args)))

    def flatten(self) -> Tensor:
        total_elements = math.prod(self.shape)

        return ops.Reshape.apply(self, new_shape=(total_elements,))
    
    def pad(self, pad_sizes: tuple[tuple[int, int], ...]) -> Tensor:
        return ops.Pad.apply(self, pad_sizes=pad_sizes)

    def slice(self, slice_sizes: tuple[tuple[int, int], ...]) -> Tensor:
        return ops.Slice.apply(self, slice_sizes=slice_sizes)
    
    def expand(self, new_shape: tuple[int, ...]) -> Tensor:
        return ops.Expand.apply(self, new_shape=new_shape)
    
    def squeeze(self, axis: Optional[int] = None):
        if axis is None: 
            return self if 1 not in self.shape else self.reshape(tuple(size for size in self.shape if size != 1))
        if axis <= 0 and self.ndim == 0: 
            return self # This is to match PyTorch behavior
        
        assert -self.ndim >= axis > self.ndim, \
            f"Dimension out of range (expected to be in range of [{-self.ndim if self.ndim > 0 else self.ndim-1}, {self.ndim-1 if self.ndim > 0 else self.ndim}], but got {axis})"
            
        if axis < 0: 
            axis += self.ndim

        return self if self.shape[axis] != 1 else self.reshape(*[size for idx, size in enumerate(self.shape) if idx != axis])

    def unsqueeze(self, axis: int):
        if axis < 0: 
            axis = self.ndim + axis + 1

        return self.reshape(self.shape[:axis] + (1,) + self.shape[axis:])

    #* Unary operations

    def neg(self) -> Tensor:
        return ops.Neg.apply(self)

    def log(self) -> Tensor:
        return ops.Log.apply(self)

    def log2(self) -> Tensor:
        return ops.Log2.apply(self)
    
    # Reduce operations

    # TODO: Mirko, 05.04.2024.
    # def reduce(fn)
    #   do common stuff for reduce ops...
    #   call fn() which could be sum, max, min

    def sum(self, axis: Optional[int | tuple[int, ...]] = None, keepdims: bool = False) -> Tensor:
        axis = tuple(i for i, _ in enumerate(self.shape)) if axis is None else axis
        axis = (axis,) if isinstance(axis, int) else axis
        axis = tuple(a if a >= 0 else len(self.shape) + a for a in axis)
        
        shape = tuple(s for i, s in enumerate(self.shape) if i not in axis)
        _sum = ops.Sum.apply(self, axis=axis)

        return _sum if keepdims else ops.Reshape.apply(_sum, new_shape=shape)
                
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        _sum = self.sum(axis, keepdims=keepdims)

        if axis is not None:
            return _sum / self.shape[axis]
        else:
            return _sum / math.prod(self.shape)

    def std(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        # Number of samples
        N = math.prod(self.shape) if axis is None else self.shape[axis]
        # Bessel's correction (taking into account the case where N = 1)
        N = max(1, N - 1)

        _mean = self.mean(axis, True)
        deviation = ((self - _mean) ** 2)
        
        return (deviation.sum(axis, keepdims) / N).sqrt()

    def var(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return self.std(axis, keepdims) ** 2
    
    # TODO: Mirko, 25.02.2024.
    # Multinomial doesn't use the replacement parameter  
    def multinomial(self, num_samples: int = 1, replacement: bool = False) -> Tensor:
        assert all(helpers.float_equal(s, 1.0) for s in self.sum(axis=-1).data), \
            "Cannot perform multinomial, Tensor rows must be probability distributions."
        return Tensor(cpp.MiniBuffer.multinomial(self._storage, num_samples, replacement), self.requires_grad)

    #* Binary operations

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
    
    def sqrt(self) -> Tensor:
        return self ** 0.5

    def matmul(self, other: Tensor) -> Tensor:
        x, y = self, other
        n1, n2 = len(x.shape), len(y.shape)
        
        assert n1 >= 1 and n2 >= 2, \
            f"Both arguments to matmul need to be at least 1D @ 2D, but they are {n1}D and {n2}D."
        assert x.shape[-1] == y.shape[-2], \
            f"Input Tensor shapes {x.shape} and {y.shape} cannot be multiplied ({x.shape[-1]} != {y.shape[-2]})"
        
        #? NOTE: Mirko, 24. 12. 2023 
        # Example to ilustrate what is happening below:
        # let x.shape = [3, 3]  &  y.shape = [3, 2]
        # x.shape = [3, 1, 3]
        # y.shape = [1, 3, 2]^T = [1, 2, 3]
        # Broadcasted: 
        # x.shape = [3, 2, 3]
        # y.shape = [3, 2, 3]
        # Performing element-wise product and then sum over the last
        # dimension (rows) is essentially giving the same result as
        # matrix multiplication.
        x = self.reshape(*self.shape[0:-1], *(1,)*min(n1-1, n2-1, 1), self.shape[-1])
        y = y.reshape(*y.shape[0:-2], *(1,)*min(n1-1, n2-1, 1), *y.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))

        return (x*y).sum(-1)

    #* Ternary functions    
    def where(self, x: Tensor, y: Tensor) -> Tensor:
        condition_, x_ = self._broadcasted(x)
        condition,  y_ = condition_._broadcasted(y)
        x, y = x_._broadcasted(y_)
        
        return ops.Where.apply(condition=condition, x=x, y=y)

    #* Activation functions
    
    def sigmoid(self) -> Tensor:
        return ops.Sigmoid.apply(self)
    
    def tanh(self) -> Tensor:
        return 2 * (2 * self).sigmoid() - 1

    def relu(self) -> Tensor:
        return ops.Relu.apply(self)
    
    def softmax(self, axis = -1) -> Tensor:
        self_exp = self.exp()

        return self_exp / self_exp.sum(axis, keepdims=True)

    #* Math functions
    
    def abs(self): 
        return self.relu() + (-self).relu()

    def sign(self): 
        return self / (self.abs() + 1e-10)

    def reciprocal(self): 
        return 1.0/self

    #* Cost functions

    def MSE(self, target: Tensor, axis: Optional[int] = None) -> Tensor:
        square_error = (self - target) ** 2
        return square_error.mean(axis)

    def cross_entropy(self, target: Tensor, axis: Optional[int] = None) -> Tensor:
        #? NOTE: Mirko, 25. 12. 2023. 
        # PyTorch uses natural log here, might be relevant later
        log_input = self.log()
        return -(target * log_input).sum(axis)

    #* Backpropagation

    def toposort(self):
        def _toposort(node: Tensor, nodes: list[Tensor]) -> list[Tensor]:
            visited = set()
            stack = [node]

            while stack:
                current_node = stack.pop()
                visited.add(current_node)

                if getattr(current_node, "_ctx", None):
                    for parent in current_node._ctx.parents:
                        if parent not in visited:
                            stack.append(parent)

                    nodes.append(current_node)

            return nodes
        
        return _toposort(self, [])

    def backward(self):
        assert self.is_scalar(), f"Backward can only be called for scalar tensors, but it has shape {self.shape})"

        self.grad = Tensor(1, requires_grad=False)
        autograd_graph = self.toposort()

        for node in autograd_graph:
            assert node.grad is not None

            grads = node._ctx.backward(node.grad._storage)
            grads = [Tensor(g, requires_grad=False) if g is not None else None
                        for g in ([grads] if len(node._ctx.parents) == 1 else grads)]
            
            for parent, grad in zip(node._ctx.parents, grads):
                if grad is not None and parent.requires_grad:
                    assert grad.shape == parent.shape, f"Grad shape must match Tensor shape, {grad.shape} != {parent.shape} ({node._ctx})"
                    parent.grad = grad if parent.grad is None else (parent.grad + grad)
            
    #* Broadcasting

    def _broadcasted(self, y: Tensor) -> tuple[Tensor, Tensor]:
        x: Tensor = self
    
        if (xshape:=x.shape) == (yshape:=y.shape):
            return (x, y)
        
        shape_delta = len(xshape) - len(yshape)

        if shape_delta > 0:
            new_shape = (1, ) * shape_delta + yshape 
            y = y.reshape(new_shape)
        elif shape_delta < 0:
            new_shape = (1, ) * -shape_delta + xshape 
            x = x.reshape(new_shape)

        if (xshape:=x.shape) == (yshape:=y.shape): 
            return (x, y)
    
        shape_ret = [max(x, y) for x, y in zip(xshape, yshape)]

        if xshape != shape_ret: 
            x = x.expand(shape_ret)
        if yshape != shape_ret: 
            y = y.expand(shape_ret)

        return (x, y)

    #* Unary operator magic methods

    def __neg__(self) -> Tensor:
        return self.neg()

    def __getitem__(self, keys) -> Tensor: # keys: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
        def normalize_int(e: int, i: int, dim_sz: int) -> int:
            if -dim_sz <= e < dim_sz: 
                return e if e != -1 else dim_sz-1
            assert False, \
                f"Index {e} is out of bounds for dimension {i} with size {self.shape[i]}"


        def normalize_slice(v, i: int, dim_sz: int) -> slice:
            if isinstance(v, slice):
                return v
            elif isinstance(v, int):
                start = normalize_int(v, i, dim_sz)
                return slice(start, start + 1)
            else:
                return slice(None)
            
        def handle_strides(sliced_tensor: Tensor, strides) -> Tensor:
            strides = tuple(abs(s) for s in strides)
            # Pad tensor
            padding = tuple((0, s - (dim_sz % s) if dim_sz % s != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape))
            padded_tensor = sliced_tensor.pad(padding)
            # Reshape tensor
            reshaped_tensor = padded_tensor.reshape(helpers.flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides)))
            new_shape = reshaped_tensor.shape[::2]
            # Slice tensor
            sliced_tensor = reshaped_tensor.slice(tuple(helpers.flatten(((0, sh), (0, 1)) for sh in new_shape)))
            return sliced_tensor

        # Convert keys to list
        orig_slices = list(keys) if isinstance(keys, tuple) else [keys]

        # Count occurrences of each type in keys
        count = defaultdict(list)
        for i, v in enumerate(orig_slices):
            count[type(v)].append(i)

        # Check number of slices and ellipsis
        num_slices = len(count[int]) + len(count[slice]) + len(count[Tensor])
        ellipsis_found = count[type(Ellipsis)]
        assert num_slices <= len(self.shape), f"Too many indices for tensor of dimension {len(self.shape)}"
        assert len(ellipsis_found) <= 1, "An index can only have a single ellipsis ('...')"

        # Replace ellipsis with appropriate number of full slices
        ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
        orig_slices[ellipsis_idx:ellipsis_idx+1] = [slice(None)] * (len(self.shape) - num_slices)

        # Normalize slices
        valid_slices = [v for v in orig_slices if v is not None]
        valid_slices = [normalize_slice(v, i, dim_sz) for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))]

        # Get start, stop, and stride for each slice
        slice_indices = [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]
        start, stop, strides = zip(*slice_indices) if slice_indices else ((), (), ())

        # Reverse slices with negative strides
        new_slice = tuple((s, e) if st > 0 else (e+1, s+1) for s, e, st in zip(start, stop, strides))

        # Slice and flip tensor
        sliced_tensor = self.slice(new_slice).flip(axis=tuple(i for i, s in enumerate(strides) if s < 0))

        # Handle strides
        if any(abs(s) != 1 for s in strides):
            sliced_tensor = handle_strides(sliced_tensor, strides)

        # Final reshape
        final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(sliced_tensor.shape), [], [], 0
        for i, s in enumerate(orig_slices):
            if s is None:
                final_shape.append(1)
            else:
                dim_shape = next(it_shape)
                if isinstance(s, int):
                    dim_collapsed += 1
                else:
                    assert isinstance(dim_shape, int), f"does not support symbolic shape {dim_shape}"
                    final_shape.append(dim_shape)
                    if isinstance(s, Tensor):
                        tensors.append(s)
                        dim.append(i - dim_collapsed)

        ret = sliced_tensor.reshape(tuple(final_shape))

        if tensors: # Fancy/tensor indexing
            # normalize idx
            idx = [t.sign().__neg__().relu() * ret.shape[d] + t for d,t in zip(dim, tensors)]
            max_dim = max(i.ndim for i in idx)
            # compute sum_dim, arange, and idx
            sum_dim = [d if n==0 else d+max_dim-n for n,d in enumerate(dim)]
            arange = [Tensor.arange(ret.shape[d], dtype=Tensor.Dtype.Int, requires_grad=False).reshape(*[1]*sd, ret.shape[d], *[1]*(ret.ndim + max_dim - n - sd - 1)) for n,(sd,d) in enumerate(zip(sum_dim, dim))]
            first_idx = [idx[0].reshape(*[1]*dim[0], *[1]*(1 + max_dim - idx[0].ndim), *idx[0].shape, *[1]*(ret.ndim - dim[0] - 1))]
            rest_idx = [i.reshape(*[1]*dim[0], *[1]*(max_dim - i.ndim), *i.shape, *[1]*(ret.ndim - dim[0] - n)) for n,i in enumerate(idx[1:], 1)]
            idx = first_idx + rest_idx
            ret = ret.reshape(*ret.shape[:sum_dim[0]+1], *[1]*max_dim, *ret.shape[sum_dim[0]+1:])

            # iteratively fancy index
            for a,i,sd in zip(arange, idx, sum_dim): 
                ret = (a==i).mul(ret).sum(sd)
                
            # special permute case
            if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1]+1)):
                ret_dims = list(range(ret.ndim))
                ret = ret.permute(ret_dims[dim[0]:dim[0]+max_dim] + ret_dims[:dim[0]] + ret_dims[dim[0]+max_dim:])

        return ret

    #* Binary operator magic methods

    def __add__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.add(other)
    
    def __radd__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.add(other, True)
    
    def __iadd__(self, other) -> Tensor: 
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.assign(self.add(other))

    def __sub__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.sub(other)

    def __rsub__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.sub(other, True)

    def __isub__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.assign(self.sub(other))

    def __mul__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.mul(other)
    
    def __rmul__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.mul(other, True)

    def __imul__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.assign(self.mul(other))

    def __truediv__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.div(other)
    
    def __rtruediv__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.div(other, True)
    
    def __itruediv__(self, other) -> Tensor: 
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.assign(self.div(other))

    def __pow__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.pow(other)

    def __rpow__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.pow(other, True)
    
    def __ipow__(self, other) -> Tensor: 
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.assign(self.pow(other))

    def __matmul__(self, other) -> Tensor:
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.matmul(other)
    
    def __imatmul__(self, other) -> Tensor: 
        if not isinstance(other, Tensor): 
            other = Tensor(other)

        return self.assign(self.matmul(other))

    def __eq__(self, other) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        x, y = self._broadcasted(other)
            
        return Tensor(x._storage == y._storage)
    
    def __ne__(self, other) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        x, y = self._broadcasted(other)
            
        return Tensor(x._storage != y._storage)
        
    def __lt__(self, other) -> Tensor: 
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        x, y = self._broadcasted(other)
            
        return Tensor(x._storage < y._storage)
    
    def __gt__(self, other) -> Tensor: 
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        x, y = self._broadcasted(other)
            
        return Tensor(x._storage > y._storage)
    
    def __le__(self, other) -> Tensor: 
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        x, y = self._broadcasted(other)
            
        return Tensor(x._storage <= y._storage)

    def __ge__(self, other) -> Tensor: 
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        x, y = self._broadcasted(other)
            
        return Tensor(x._storage >= y._storage)

    def __hash__(self) -> int:
        return id(self)

    #* Data handlers

    # TODO: Mirko, 05.02.2024
    # Switching between 'train' and 'eval' mode was not compati-
    # ble with the previous version of this function. That is
    # why the lines are commented out. Find a better way to set
    # the requires_grad field.
    #? NOTE: Mirko, 24. 12. 2023
    # This function shall be used whenever a Tensor needs to be
    # modified inplace. It is essentially a shallow copy.
    # Other use case for this function is when traversing a 
    # collection of tensors (i.e. list of Tensors that repre-
    # sent parameters of an NN module) and needing to update
    # the Tensors from there.
    def assign(self, x) -> Tensor:
        if not isinstance(x, Tensor): 
            x = Tensor(x)
        assert self.shape == x.shape, f"Assign shape mismatch {self.shape} != {x.shape}."
        # assert not x.requires_grad

        self._storage = x._storage
        # TODO: Mirko, 04.05.2024.
        # Temporary hack since it's sometimes necessary to create a
        # copy of a Tensor (including the ._ctx and grad fields) in
        # order to perform some operations without leaving the auto-
        # grad graph.
        #self._ctx = x._ctx
        #self.grad = x.grad
        
        self.requires_grad = x.requires_grad
        
        return self

    #? NOTE: Mirko, 24. 12. 2023
    # This function shall be used to detach the Tensor from
    # the autograd graph. It is mostly used in the NN modules
    # to use data from the Tensors without adding those operations
    # to the graph.
    def detach(self) -> Tensor: 
        return Tensor(self._storage)

    #* Utility
    
    def eq(self, other: Tensor) -> bool:
        return self._storage.eq(other._storage)
    
    def all_close(self, other: Tensor, equal_nan: bool = False) -> bool:
        return self._storage.all_close(other._storage, equal_nan=equal_nan)

    def is_close(self, other: Tensor, equal_nan: bool = False) -> Tensor:
        return Tensor(self._storage.is_close(other._storage, equal_nan=equal_nan), dtype=Tensor.Dtype.Bool)
    
    def is_scalar(self) -> bool:
        return self.size == 1
    
    def is_square(self) -> bool:
        assert len(self.shape) >= 2, f"Cannot check for squareness on a {len(self.shape)}D Tensor. Expected 2D or higher."
        return self._storage.is_square()
    
    def item(self) -> Value:
        assert self.is_scalar(), f"a Tensor with {len(self.size)} elements cannot be converted to Scalar."
        return self._storage.item()

    def num_el(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"<Tensor: \n{self._np} with grad {self.grad._np if self.grad else None}>"
    
    def __str__(self) -> str:
        return f"<Tensor: \n{self._np} with grad {self.grad._np if self.grad else None}>"