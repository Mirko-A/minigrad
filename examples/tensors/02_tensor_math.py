# This file contains a list of different mathematical operations that can be
# performed on Tensor objects.

# NOTE: 
# All of the methods which create Tensor objects have an additional boolean parameter
# 'requires_grad'. This parameter determines whether a computation graph will be created
# from mathematical operations performed on those Tensors. In case the 'requires_grad'
# parameter is set to True, the graph will be created and can be later used to calculate
# the gradients of all Tensors which participate in the mathematical operations by calling
# the backward() method of a Tensor which is an output of a certain operation.
# More on this here: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html 

# NOTE:
# All of the binary Tensor operations support broadcasting. Broadcasting can be performed either
# when performing the operation with another Tensor or with a scalar (int or float) value.
# More on broadcasting here: https://numpy.org/doc/stable//user/basics.broadcasting.html

from minitorch.tensor import Tensor

# Unary operations

t0 = Tensor.arange(0, 9, requires_grad=True).reshape([3, 3])
t1 = Tensor.arange(0, 9, requires_grad=True).reshape([3, 3])
# print(t0)
# print(t1)
t0 = -t0        # Negate entire Tensor
# print(t0)
t0 = t0.log()   # Natural log
# print(t0)
t1 = t1.log2()  # Binary log
# print(t1)

# Binary operations

t2 = t1 * 2     # Scalar multiplication*.
# print(t2)
t3 = t0 + t2    # Element-wise Tensor addition*.
# print(t3)
t4 = t3 ** 0.5  # Element-wise Tensor exponentiation*. 
                # Sqrt can also be done like this: t4 = t3.sqrt().
# print(t4)
t5 = t2 @ t4    # Matrix multiplication. Usual rules apply here.
# print(t5)

# Activation functions

t6 = t0.sigmoid()
# print(t6)
t7 = t1.tanh()
# print(t7)
t8 = t5.relu()
# print(t8)
t9 = t7.softmax()
# print(t9)

t10 = t6 + t7 ** t8
# print(t10)
t11 = t10.sum()
# print(t11)

t11.backward()
# print(t11.grad)
# print(t0.grad)
# print(t1.grad)
# print(t2.grad)
# print(t3.grad)
# print(t4.grad)
# print(t5.grad)
# print(t6.grad)
# print(t7.grad)
# print(t8.grad)
# print(t9.grad)
# print(t10.grad)

# NOTE:
# Loss functions like Mean Squared Error and Categorical Cross 
# Entropy are also supported but more on these in nn examples.