from matrix import Matrix
import torch
import torch.nn as nn
import math
import time

m = Matrix.from_2d_array([[43.6, 44.4, 45.2, 46.0, 46.8], [32.4, 31.9, 28.6, 39.1, 40.2], [40.8, 52.1, 33.8, 51.7, 60.3]])
target_1 = Matrix.from_2d_array([[41.0, 30.0, 42.0], [45.0, 35.0, 50.0], [49.0, 28.0, 35.0], [47.0, 37.0, 54.0], [44.0, 39.0, 59.0]])
target_2 = Matrix.from_2d_array([[41.0, 45.0, 49.0, 47.0, 44.0], [30.0, 35.0, 28.0, 37.0, 39.0], [42.0, 50.0, 35.0, 54.0, 59.0]])

mse_1 = m.mse(target_1)
mse_2 = m.mse(target_2, 1)

print(f'MSE 1:\n{mse_1}\n\nMSE 2:\n{mse_2}')

# start_time = time.time()

# N = 16

# a = torch.randn(N, N)
# b = torch.randn(N, N)
# a = Matrix.randn(N, N)
# b = Matrix.randn(N, N)

# for i in range(100):
    # c = torch.matmul(a, b)
    # c = Matrix.matmul(a, b)

# print("--- %s seconds ---" % (time.time() - start_time))

# a = Matrix.fill(8, 8, 1.0)
# b = Matrix.tril(a)
# c = Matrix.replace(Matrix.replace(b, 0.0, -math.inf), 1.0, 0.0)
# print(c.softmax(1))

# m = Matrix.from_2d_array([[0.7, 2.1], [0.2, 4.1], [2.3, 1.7]])
# b = 2.0 * m
# c = b + m
# n = c.softmax(0)
# n_1 = c.softmax(1)
# a = n.sum()
# x = a.exp()
# x.backward()
# print(x)
# print(m.grad())

# target_mat_0 = Matrix.from_2d_array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
# target_mat_1 = Matrix.from_2d_array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
# ce_0 = n.cross_entropy(target_mat_0, 0)
# ce_1 = n_1.cross_entropy(target_mat_1, 1)

# print(f'Softmax (0):\n{n}\nCross-entropy (0):\n{ce_0}\n\nSoftmax (1):\n{n_1}\nCross-entropy (1):\n{ce_1}\n')


# #n = Matrix.from_2d_array([[1.8, -2.1], [-0.3, 0.3]])
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

# m_pt = torch.tensor([[0.7, 2.1], [0.2, 4.1], [2.3, 1.7]]); m_pt.requires_grad = True
#print(torch.tril(m_pt))
# b_pt = 2 * m_pt
# c_pt = b_pt + m_pt
# n_pt = c_pt.softmax(0)
# a_pt = n_pt.sum()
# x_pt = a_pt.exp()
# x_pt.backward()

# loss = nn.CrossEntropyLoss()
# test_1 = torch.tensor([c_pt[0][0], c_pt[1][0], c_pt[2][0]])
# test_2 = torch.tensor([c_pt[0][1], c_pt[1][1], c_pt[2][1]])
# test_target_1 = torch.tensor([0.0, 0.0, 1.0])
# test_target_2 = torch.tensor([0.0, 1.0, 0.0])

# ce_pt_1 = loss(test_1, test_target_1)
# ce_pt_2 = loss(test_2, test_target_2)

# print(f'PT Softmax (0):\n{test_1}\nPT Cross-entropy (0):\n{ce_pt_1}\n\nPT Softmax (1):\n{test_2}\nPT Cross-entropy (1):\n{ce_pt_2}')

# print(x_pt)
# print(m_pt.grad)
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

# print(type(True))


