# This file contains a list of different options the user of minitorch
# has to create Tensor objects. Currently, minitorch supports Tensors
# up to rank 4.

from minitorch.tensor import Tensor

# First let's see how we can create new Tensor objects.

t0 = Tensor(2) # Creates a scalar Tensor which holds the value 2.
# print(t0) 
# print(t0.shape) 
# print(t0.is_scalar()) 

t1 = Tensor.fill([2, 3], 1.0) # Creates a 2x3 Tensor filled with ones.
# print(t1)
# print(t1.shape)
t2 = Tensor.ones([2,3]) # Alternatively we can use the 'ones' (or 'zeros') method.
# print(t2)
# print(t2.shape)

t3 = Tensor.arange(0, 9) # Creates a Tensor which holds values from 0 (inclusive) to 9 (exclusive).
# print(t3)
# print(t3.shape)

t4 = Tensor.randn([2, 3, 3], 0.0, 1.0) # Creates a 2x2x3 Tensor filled with random values such that
                                       # their mean is 0.0 and their standard deviation is 1.0.
# print(t4)
# print(t4.shape)

t5 = Tensor.uniform([2, 3, 3], -2.0, 2.0) # Creates a 2x2x3 Tensor filled with uniformly distributed
                                          # random values between -2.0 and 2.0.
# print(t5)
# print(t5.shape)

# Now let's see how we can use existing Tensors to create new ones.

# print(t1)
t1 = Tensor.replace(t1, 1.0, 0.5) # Replace all occurances of 1.0 with 0.5 in the provided Tensor.
# print(t1)

mask = [True, True, False, False, False, True]
# print(t2)
t2 = Tensor.masked_fill(t2, mask, 3.5) # Fill the provided Tensor with 3.5 but only in the places
                                       # where mask's value is 'True'. Note that the mask needs to
                                       # be a list of booleans since the data stored inside a Tensor
                                       # is also in the form of a list. Usually you wouldn't construct
                                       # these lists by hand, but could instead use the ==, < or > op-
                                       # erators to do it for you (essentially a tool to replace values).
# print(t2)

t5 = Tensor.tril(t5, diagonal=0) # Create a triangular matrix* Tensor with the diagonal parameter being
                                 # the offset from which the triangle starts. If the offset is 0, the
                                 # triangular Matrix is formed below the main diagonal.
# print(t5)

t6 = Tensor.concat(0, t4, t5) # Concatenate the provided Tensors along axis 0. The shape of all provided
                              # Tensors must match along all axes except the one they're being concatenated
                              # along. Can also be done like this: t4.cat(0, t5).
# print(t6)

# *in a triangular matrix, all values above the diagonal are set to 0

# All of the methods which create Tensor objects have an additional boolean parameter
# 'requires_grad'. This parameter determines whether a computation grad will be created
# from mathematical operations performed on those Tensors. In case the 'requires_grad'
# parameter is set to True, the graph will be created and can be later used to calculate
# the gradients of all Tensors which participate in the mathematical operations by calling
# the backward() method of a Tensor which is an output of a certain operation.
# More on this here: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html 
