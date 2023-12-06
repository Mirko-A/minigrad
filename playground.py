import minigrad
import torch

m = minigrad.Matrix.from_2d_array([[0.7,  2.1], [0.2,  4.1],  [2.3, 1.7]])
b = 2.0 * m
c = b + m
n = c.softmax(0)
a = n.sum()
x = a.exp()
x.backward()
print(x)
print(m.grad())

# #n = minigrad.Matrix.from_2d_array([[1.8, -2.1], [-0.3, 0.3]])
# n = 2.0 * m
# a = n.sum(0)
# b = a.sum()
# c = b / 3.7
# d = c * 0.3
# e = d.exp()
# print(e)
# e.backward()

# for row in m.data:
#     for value in row:
#         print(value.grad)

m_pt = torch.tensor([[0.7,  2.1], [0.2,  4.1],  [2.3, 1.7]]); m_pt.requires_grad = True
b_pt = 2 * m_pt
c_pt = b_pt + m_pt
n_pt = c_pt.softmax(0)
a_pt = n_pt.sum()
x_pt = a_pt.exp()
x_pt.backward()
print(x_pt)
print(m_pt.grad)
# n_pt = 2.0 * m_pt
# a_pt = n_pt.sum(0)
# b_pt = a_pt.sum()
# c_pt = b_pt / 3.7
# d_pt = c_pt * 0.3
# e_pt = d_pt.exp()
# print(e_pt)
# e_pt.backward()
# print(m_pt.grad)
# #n_pt = torch.tensor([[1.8, -2.1], [-0.3, 0.3]]); m_pt.requires_grad = True
