from minitorch.tensor import Tensor
from minitorch.nn.module import Linear

# Read your data
x = ... Tensor()

layer = Linear(...)

y = layer(x)
print(y)