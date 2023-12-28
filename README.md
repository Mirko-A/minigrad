# minitorch

---

Minitorch is a small-scale attempt to replicate a subset of the PyTorch framework. It is (being) developed for educational purposes. It is written in Python and the current version uses no third-party libraries whatsoever.

### Repository structure

| Component            | Description                       |
| -------------------- | ----------------------------------|
| minitorch.tensor     | Tensor library                    |
| minitorch.nn.module  | Neural network library            |
| minitorch.nn.optim   | Optimizer modules                 |
| examples             | Examples of how to use Minitorch  |

### Project setup

```Console
> git clone https://github.com/Mirko-A/minitorch
> pip install -e minitorch
```

### Examples

Here are a few examples of how one would use Minitorch. For more detailed ones, check out the *examples* directory.

##### Working with Tensors

```Python
from minitorch.tensor import Tensor

t0 = Tensor([[[0, 1, 2],   # Creates a 2x2x3 Tensor.
              [3, 4, 5]],
             [[0, 1, 2],
              [3, 4, 5]]])

t1 = Tensor.fill((2, 2, 3), 0.5)   # Creates a 2x2x3 Tensor 
                                   # filled with 0.5.

t2 = Tensor.randn((2, 3, 2), 0.0, 2.0)  # Creates a 2x3x2 Tensor filled with
                                        # filled with random values where 
                                        # mean = 0.0, std_dev = 2.0 (optional)

x = t0 + 1.3  # Addition* with a scalar
y = t0 + t1   # Elementwise addition*
z = t0 @ t1   # Matrix multiplication
```
<sup>*(or subtraction, multiplication, division)</sup>

##### ML Framework
```Python
from minitorch.nn.module import Linear, Sequence, Sigmoid, MSELoss
from minitorch.nn.optim import Adam

# Read your data
inputs = ...
targets = ...

seq_net = Sequence(
    Linear(4, 4),
    Sigmoid(),
    Linear(4, 1),
    Sigmoid()
)

mse = MSELoss()
adam = Adam(seq_net.params(), 0.05) # 0.05 is the learning rate

for _ in (range(100))
    pred = seq_net(inputs)
    loss = mse(inputs, targets)

    loss.backward()
    adam.step()
    adam.zero_grad()

```


### Inspiration

Minitorch is inspired by the following two projects:

- [PyTorch](https://github.com/pytorch/pytorch) – for the theoretical part
- [Tinygrad](https://github.com/tinygrad/tinygrad) – for the implementation