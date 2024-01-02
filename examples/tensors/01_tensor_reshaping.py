# This file contains a list of different options the user of minitorch
# has to reshape and mutate existing Tensor objects to create new ones
# which better suit the given use case.

from minitorch.tensor import Tensor

t0 = Tensor.arange(0, 9)
# print(t0)
t0 = t0.reshape([3, 3]) # Reshape an existing Tensor to have a 3x3 shape but keep the same data (range [0-9))
# print(t0)

t1 = Tensor([[2, 3, 4],
             [2, 3, 4]])
# print(t1)
t1 = t1.flatten()   # Flatten a 2x3 Tensor to have a 1x6 shape
# print(t1)

t2 = Tensor.arange(0, 9).reshape((3, 3))
# print(t2)
t2 = t2.permute((1, 0)) # Swap the dimensions to have the order 1, 0 (instead of 0, 1). In this
                        # case it boils down to transposing a 3x3 matrix.
# print(t2)
t2 = t2.transpose() # Another way to transpose a Tensor along the two given axes. It is a special case
                    # of the permute operation. Default values for the two axes are -2 and -1 which 
                    # performs the matrix transpose operation.
# print(t2)

t3 = Tensor.randn([2, 3])
# print(t3)
t3 = t3.pad(1, (1, 2)) # Pad the Tensor with zeros along the 1st axis. There will be 
                       # one zero emplaced before the axis and two zeros appended to
                       # the end.
# print(t3)

t4 = Tensor.randn([4, 4])
# print(t4)
t4 = t4.shrink(0, (2, 0)) # Shrink the Tensor by removing two values from the start
                            # and zero values from the end of a given axis (0 in this case).
# print(t4)

t5 = Tensor.arange(0, 6).reshape((1, 1, 6)) # A 1x1x6 Tensor
# print(t5)
t5 = t5.expand([3, 4, 6]) # Expand along the 0th and 1st axis to be 3x3x6. Tensors can only be expanded
                         # along singular axes.
# print(t5)
