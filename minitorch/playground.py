from minitorch.nn.linear import Linear
from minitorch.nn.optim import SGD
from minitorch.tensor import Tensor

import numpy as np

def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))

def main():
    a = Tensor([[[[1.0, 2.0],
                 [3.0, 4.0]],
                [[1.0, 2.0],
                 [3.0, 2.0]]],
                 [[[1.0, 2.0],
                 [3.0, 4.0]],
                [[1.0, 2.0],
                 [3.0, 2.0]]]], requires_grad=True)
    b = Tensor([[1, 2, 3],
                [1, 2, 3]], requires_grad=True)


    print(b)
    print(b.shape)
    print(b.data.strides)
    print(b.permute((1, 0)))
    # c = a + b
    # c1 = c / 1.7
    # c2 = c1.exp()
    # c_t = a_t + b_t ; c_t.retain_grad()
    # c1_t = c_t / 1.7 ; c1_t.retain_grad()
    # c2_t = c1_t.exp() ; c2_t.retain_grad()

    # d = c2.T() 
    # d_t = c2_t.transpose(0, 1) ; d_t.retain_grad()

    # e = d.sum()
    # e_t = torch.sum(d_t) ; e_t.retain_grad()

    # e.backward()
    # e_t.backward()

    # print(a.grad)
    # print(a_t.grad)

if __name__ == "__main__":
    main()