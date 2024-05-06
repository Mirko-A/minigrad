import unittest
import math

from minitorch.tensor import Tensor

class TestTensorGenerationFuncs(unittest.TestCase):
    def test_tensor_full(self):
        a = Tensor.full((3, 2), 1.3)

        # Check dims
        self.assertEqual(a.shape[0], 3)
        self.assertEqual(a.shape[1], 2)

        # Check data
        self.assertTrue(all((a == 1.3)._np.flatten().tolist()))

    def test_tensor_zeros(self):
        a = Tensor.zeros((2, 4))

        # Check dims
        self.assertEqual(a.shape[0], 2)
        self.assertEqual(a.shape[1], 4)

        # Check data
        self.assertTrue(all((a == 0.0)._np.flatten().tolist()))

    def test_tensor_randn(self):
        a = Tensor.randn((5, 1))

        # Check dims
        self.assertEqual(a.shape[0], 5)
        self.assertEqual(a.shape[1], 1)

        # Impossible to check random data 
    
    def test_tensor_masked_fill(self):
        old = Tensor([[ 0.7,  3.2, 1.1], 
                      [ 3.2, -3.9, 0.2], 
                      [-1.5,  3.2, 3.2]])

        new = old.masked_fill(old == 3.2, 1.7)

        # Check dims
        self.assertEqual(new.shape[0], 3)
        self.assertEqual(new.shape[1], 3)
        
        # Check data
        for o, n in zip(old._np.flatten().tolist(), new._np.flatten().tolist()):
            if o == 3.2:
                self.assertEqual(n, 1.7)

    def test_tensor_replace(self):
        old = Tensor([[ 0.7,  3.2, 1.1], 
                      [ 3.2, -3.9, 0.2], 
                      [-1.5,  3.2, 3.2]])

        new = old.replace(3.2, 1.7)

        # Check dims
        self.assertEqual(new.shape[0], 3)
        self.assertEqual(new.shape[1], 3)

        # Check data
        for o, n in zip(old._np.flatten().tolist(), new._np.flatten().tolist()):
            if o == 3.2:
                self.assertEqual(n, 1.7)

    def test_tensor_tril_main_diagonal(self):
        old = Tensor([[ 0.7,  3.2, 1.1], 
                      [ 3.2, -3.9, 0.2], 
                      [-1.5,  3.2, 3.2]])

        new = Tensor.tril(old)
        expected_new = Tensor([[ 0.7,  0.0, 0.0], 
                               [ 3.2, -3.9, 0.0], 
                               [-1.5,  3.2, 3.2]])

        # Check dims
        self.assertEqual(new.shape[0], 3)
        self.assertEqual(new.shape[1], 3)
        
        # Check data
        for new_row, expected_new_row in zip(new.data, expected_new.data):
            for new_value, expected_new_value in zip(new_row, expected_new_row):
                self.assertAlmostEqual(new_value, expected_new_value)

#     def test_tensor_tril_anti_diagonal(self):
#         old = Tensor([[ 0.7,  3.2, 1.1], 
#                       [ 3.2, -3.9, 0.2], 
#                       [-1.5,  3.2, 3.2]])

#         new = Tensor.tril(old, Tensor.Diagonal.ANTI)
#         expected_new = Tensor([[0.7,  3.2, 1.1], 
#                                [0.0, -3.9, 0.2], 
#                                [0.0,  0.0, 3.2]])
#         # Check dims
#         self.assertEqual(new.shape[0], 3)
#         self.assertEqual(new.shape[1], 3)
        
#         # Check data
#         for new_row, expected_new_row in zip(new.data, expected_new.data):
#             for new_value, expected_new_value in zip(new_row, expected_new_row):
#                 self.assertAlmostEqual(new_value, expected_new_value)

# class TestBinaryOps(unittest.TestCase):
#     def test_tensor_is_equal_to_tensor(self):
#         a = Tensor([[3.0,  2.4, -1.8], 
#                     [1.2, -0.5,  2.3], 
#                     [2.1,  1.4,  0.8]])
        
#         b = Tensor([[3.0,  2.4, -1.8], 
#                     [1.2, -0.5,  2.3], 
#                     [2.1,  1.4,  0.8]])
        
#         self.assertEqual(a == b, True)
        
#     def test_tensor_is_elementwise_equal_to_float(self):
#         a = Tensor([[ 0.7,  3.2, 1.1],  
#                     [-1.5,  3.2, 3.2]])
        
#         target = 3.2

#         result = a == target
#         expected_result = [[False, True , False],
#                            [False, True , True ],]
        
#         # Check result
#         for res_row, exp_res_row in zip(result, expected_result):
#             for res_val, exp_res_val in zip(res_row, exp_res_row):
#                 self.assertEqual(res_val, exp_res_val) 

#     def test_add_two_tensor_objects(self):
#         a = Tensor([[3.0,  2.4, -1.8], 
#                     [1.2, -0.5,  2.3], 
#                     [2.1,  1.4,  0.8]])
        
#         b = Tensor([[ 0.8,  1.4, 2.1], 
#                     [ 2.3, -0.5, 1.2], 
#                     [-1.8,  2.4, 3.0]])
        
#         result = a + b

#         # Check result
#         expected_result = Tensor([[ 3.8,  3.8, 0.3], 
#                                   [ 3.5, -1.0, 3.5], 
#                                   [ 0.3,  3.8, 3.8]])

#         rows, cols = result.shape[0], result.shape[1]
#         for row in range(rows):
#             for col in range(cols):
#                 self.assertAlmostEqual(result[row][col], expected_result[row][col])

#     def test_multiply_tensor_and_float(self):
#         a = Tensor([[3.0,  2.4, -1.8], 
#                     [1.2, -0.5,  2.3], 
#                     [2.1,  1.4,  0.8]])
        
#         b = 1.7

#         result = a * b

#         # Check result
#         expected_result = Tensor([[3.0 * 1.7,  2.4 * 1.7, -1.8 * 1.7], 
#                                   [1.2 * 1.7, -0.5 * 1.7,  2.3 * 1.7], 
#                                   [2.1 * 1.7,  1.4 * 1.7,  0.8 * 1.7]])

#         rows, cols = result.shape[0], result.shape[1]
#         for row in range(rows):
#             for col in range(cols):
#                 self.assertAlmostEqual(result[row][col], expected_result[row][col])

#     def test_multiply_float_and_tensor(self):
#         a = Tensor([[3.0,  2.4, -1.8], 
#                                   [1.2, -0.5,  2.3], 
#                                   [2.1,  1.4,  0.8]])
        
#         b = 1.7

#         result = b * a

#         # Check result
#         expected_result = Tensor([[3.0 * 1.7,  2.4 * 1.7, -1.8 * 1.7], 
#                                   [1.2 * 1.7, -0.5 * 1.7,  2.3 * 1.7], 
#                                   [2.1 * 1.7,  1.4 * 1.7,  0.8 * 1.7]])

#         rows, cols = result.shape[0], result.shape[1]
#         for row in range(rows):
#             for col in range(cols):
#                 self.assertAlmostEqual(result[row][col], expected_result[row][col])

#     def test_divide_tensor_and_float(self):
#         a = Tensor([[3.0,  2.4, -1.8], 
#                     [1.2, -0.5,  2.3], 
#                     [2.1,  1.4,  0.8]])
        
#         b = 1.7

#         result = a / b

#         # Check result
#         expected_result = Tensor([[3.0 / 1.7,  2.4 / 1.7, -1.8 / 1.7], 
#                                   [1.2 / 1.7, -0.5 / 1.7,  2.3 / 1.7], 
#                                   [2.1 / 1.7,  1.4 / 1.7,  0.8 / 1.7]])

#         rows, cols = result.shape[0], result.shape[1]
#         for row in range(rows):
#             for col in range(cols):
#                 self.assertAlmostEqual(result[row][col], expected_result[row][col])

#     def test_tensor_matmul_product(self):
#         a = Tensor([[3.0,  2.4, -1.8], 
#                     [1.2, -0.5,  2.3], 
#                     [2.1,  1.4,  0.8]])
        
#         b = Tensor([[0.8],  [1.4], [2.1]])
        
#         result = a.matmul(b)
        
#         # Check shape
#         self.assertEqual(a.shape[0], result.shape[0])
#         self.assertEqual(b.shape[1], result.shape[1])

#         # Check result
#         expected_result = Tensor([[1.98], [5.09], [5.32]])

#         rows, cols = result.shape[0], result.shape[1]
#         for row in range(rows):
#             for col in range(cols):
#                 self.assertAlmostEqual(result[row][col], expected_result[row][col])

# class TestUnaryOps(unittest.TestCase):
#     def test_tensor_transpose(self):
#         a = Tensor([[3.0,  2.4, -1.8], 
#                     [1.2, -0.5,  2.3]])
        
#         result = a.T

#         # Check shape
#         self.assertEqual(a.shape[0], result.shape[1])
#         self.assertEqual(a.shape[1], result.shape[0])

#         # Check result
#         expected_result = Tensor([[ 3.0,  1.2],
#                                   [ 2.4, -0.5],
#                                   [-1.8,  2.3]])

#         rows, cols = result.shape[0], result.shape[1]
#         for row in range(rows):
#             for col in range(cols):
#                 self.assertAlmostEqual(result[row][col], expected_result[row][col])

#     def test_tensor_flatten(self):
#         old = Tensor([[ 0.7,  3.2, 1.1], 
#                       [ 3.2, -3.9, 0.2], 
#                       [-1.5,  3.2, 3.2]])
        
#         new = old.flatten()
#         print(new.shape)
#         expected_new = Tensor([0.7,  3.2, 1.1, 3.2, -3.9, 0.2, -1.5,  3.2, 3.2])

#         # Check dims
#         self.assertEqual(new.shape[0], (old.shape[0] * old.shape[1]))
        
#         # Check data
#         for new_row, expected_new_row in zip(new.data, expected_new.data):
#             for new_value, expected_new_value in zip(new_row, expected_new_row):
#                 self.assertAlmostEqual(new_value, expected_new_value)

#     def test_tensor_sum_dim_not_specified(self):
#         old = Tensor([[ 0.7,  3.2, 1.1], 
#                       [ 3.2, -3.9, 0.2], 
#                       [-1.5,  3.2, 3.2]])
        
#         result = old.sum()
#         expected_result = sum([*old.flatten().data])

#         # Check dims
#         self.assertEqual(result.shape[0], 1)
        
#         # Check data
#         self.assertAlmostEqual(result.item(), expected_result, delta=0.00001)

#     def test_tensor_sum_dim_0(self):
#         old = Tensor([[ 0.7,  3.2, 1.1], 
#                       [ 3.2, -3.9, 0.2], 
#                       [-1.5,  3.2, 3.2]])
        
#         result = old.sum(axis=0)
#         expected_result = Tensor([sum(row) for row in old.T.data])

#         # Check dims
#         self.assertEqual(result.shape[0], 3)
#         self.assertEqual(result.shape[1], 1)
        
#         # Check data
#         for result_row, expected_result_row in zip(result.data, expected_result.data):
#             for result_value, expected_result_value in zip(result_row, expected_result_row):
#                 self.assertAlmostEqual(result_value, expected_result_value)

#     def test_tensor_sum_dim_1(self):
#         data = [[ 0.7,  3.2, 1.1], 
#                 [ 3.2, -3.9, 0.2], 
#                 [-1.5,  3.2, 3.2]]
#         old = Tensor(data)
        
#         result = old.sum(axis=1)
#         expected_result = Tensor([sum(row) for row in data])

#         # Check dims
#         self.assertEqual(result.shape[0], 3)
        
#         # Check data
#         for result_row, expected_result_row in zip([result.data], [expected_result.data]):
#             for result_value, expected_result_value in zip(result_row, expected_result_row):
#                 self.assertAlmostEqual(result_value, expected_result_value, delta=0.000001)

#     def test_tensor_exp(self):
#         input = Tensor([[ 0.7,  3.2, 1.1], 
#                         [ 3.2, -3.9, 0.2], 
#                         [-1.5,  3.2, 3.2]])
        
#         result = input.exp()

#         # Check dims
#         self.assertEqual(result.shape[0], input.shape[0])
#         self.assertEqual(result.shape[1], input.shape[1])
        
#         # Check data
#         for input_row, result_row in zip(input.data, result.data):
#             for input_value, result_value in zip(input_row, result_row):
#                 self.assertAlmostEqual(math.e ** input_value, result_value)

#     def test_tensor_log_base_2(self):
#         input = Tensor([[0.7, 3.2, 1.1], 
#                                       [3.2, 3.9, 0.2], 
#                                       [1.5, 3.2, 3.2]])
        
#         result = input.log(2)

#         # Check dims
#         self.assertEqual(result.shape[0], input.shape[0])
#         self.assertEqual(result.shape[1], input.shape[1])
        
#         # Check data
#         for input_row, result_row in zip(input.data, result.data):
#             for input_value, result_value in zip(input_row, result_row):
#                 self.assertAlmostEqual(math.log(input_value, 2), result_value)

#     def test_tensor_log_base_e(self):
#         input = Tensor([[0.7, 3.2, 1.1], 
#                                       [3.2, 3.9, 0.2], 
#                                       [1.5, 3.2, 3.2]])
        
#         result = input.log(math.e)

#         # Check dims
#         self.assertEqual(result.shape[0], input.shape[0])
#         self.assertEqual(result.shape[1], input.shape[1])
        
#         # Check data
#         for input_row, result_row in zip(input.data, result.data):
#             for input_value, result_value in zip(input_row, result_row):
#                 self.assertAlmostEqual(math.log(input_value, math.e), result_value)

#     def test_tensor_sigmoid(self):
#         input = Tensor([[ 0.7,  3.2, 1.1], 
#                         [ 3.2, -3.9, 0.2], 
#                         [-1.5,  3.2, 3.2]])
        
#         result = input.sigmoid()
#         expected_result = Tensor([[ 0.66818777, 0.96083427, 0.750260105], 
#                                   [ 0.96083427, 0.01984030, 0.549833997], 
#                                   [ 0.18242552, 0.96083427, 0.96083427]])

#         # Check dims
#         self.assertEqual(result.shape[0], input.shape[0])
#         self.assertEqual(result.shape[1], input.shape[1])
        
#         # Check data
#         for result_row, expected_result_row in zip(result.data, expected_result.data):
#             for result_value, expected_result_value in zip(result_row, expected_result_row):
#                 self.assertAlmostEqual(result_value, expected_result_value)

#     def test_tensor_relu(self):
#         input = Tensor([[ 0.7,  3.2, 1.1], 
#                         [ 3.2, -3.9, 0.2], 
#                         [-1.5,  3.2, 3.2]])
        
#         result = input.relu()
#         expected_result = Tensor([[ 0.7,  3.2, 1.1], 
#                                   [ 3.2,  0.0, 0.2], 
#                                   [ 0.0,  3.2, 3.2]])

#         # Check dims
#         self.assertEqual(result.shape[0], input.shape[0])
#         self.assertEqual(result.shape[1], input.shape[1])
        
#         # Check data
#         for result_row, expected_result_row in zip(result.data, expected_result.data):
#             for result_value, expected_result_value in zip(result_row, expected_result_row):
#                 self.assertAlmostEqual(result_value, expected_result_value)

    # def test_tensor_tanh(self):
    #     input = tensor([[ 0.7,  3.2, 1.1], 
    #                                   [ 3.2, -3.9, 0.2], 
    #                                   [-1.5,  3.2, 3.2]])
        
    #     result = input.tanh()

    #     # Check dims
    #     self.assertEqual(result.shape[0], input.shape[0])
    #     self.assertEqual(result.shape[1], input.shape[1])
        
    #     # Check data
    #     for input_row, result_row in zip(input.data, result.data):
    #         for input_value, result_value in zip(input_row, result_row):
    #             self.assertAlmostEqual(input_value.tanh().data, result_value.data)

    # def test_tensor_softmax_dim_0(self):
    #     input = tensor([[ 0.7,  3.2, 1.1], 
    #                                   [ 3.2, -3.9, 0.2], 
    #                                   [-1.5,  3.2, 3.2]])
        
    #     result = input.softmax(dim=0)
    #     expected_result = tensor([[0.0752258819, 0.4997938080, 0.104463303],
    #                                             [0.9164388527, 0.0004123823, 0.0424716097],
    #                                             [0.0083352653, 0.4997938088, 0.8530650866]])
    #     # Check dims
    #     self.assertEqual(result.shape[0], input.shape[0])
    #     self.assertEqual(result.shape[1], input.shape[1])
        
    #     # Check data
    #     for result_row, expected_row in zip(result.data, expected_result.data):
    #         for result_value, expected_value in zip(result_row, expected_row):
    #             self.assertAlmostEqual(result_value.data, expected_value.data)

    # def test_tensor_softmax_dim_1(self):
    #     input = tensor([[ 0.7,  3.2, 1.1], 
    #                                   [ 3.2, -3.9, 0.2], 
    #                                   [-1.5,  3.2, 3.2]])
        
    #     result = input.softmax(dim=1)
    #     expected_result = tensor([[0.06814626, 0.83019145, 0.101662280],
    #                                             [0.951826016, 7.8536e-04, 0.0473886269],
    #                                             [4.5271e-03, 0.497736474, 0.4977364744]])
    #     # Check dims
    #     self.assertEqual(result.shape[0], input.shape[0])
    #     self.assertEqual(result.shape[1], input.shape[1])
        
    #     # Check data
    #     for result_row, expected_row in zip(result.data, expected_result.data):
    #         for result_value, expected_value in zip(result_row, expected_row):
    #             self.assertAlmostEqual(result_value.data, expected_value.data)

    # def test_tensor_softmax_dim_1(self):
    #     input = tensor([[ 0.7,  3.2, 1.1], 
    #                                   [ 3.2, -3.9, 0.2], 
    #                                   [-1.5,  3.2, 3.2]])
        
    #     result = input.softmax(dim=1)
    #     expected_result = tensor([[0.06814626, 0.83019145, 0.101662280],
    #                                             [0.951826016, 7.8536e-04, 0.0473886269],
    #                                             [4.5271e-03, 0.497736474, 0.4977364744]])
        
    #     # Check dims
    #     self.assertEqual(result.shape[0], input.shape[0])
    #     self.assertEqual(result.shape[1], input.shape[1])
        
    #     # Check data
    #     for result_row, expected_row in zip(result.data, expected_result.data):
    #         for result_value, expected_value in zip(result_row, expected_row):
    #             self.assertAlmostEqual(result_value.data, expected_value.data)

    # def test_tensor_cross_entropy(self):
    #     input = tensor([[0.7, 0.2, 0.1], 
    #                                   [0.5, 0.25, 0.25], 
    #                                   [0.95, 0.01, 0.04]])

    #     target = tensor([[0, 1, 0], 
    #                                    [1, 0, 0], 
    #                                    [1, 0, 0]])
        
    #     result = input.cross_entropy(target)
    #     expected_result = tensor.from_1d_array([2.321928094, 1.0000, 0.07400058144])
        
    #     # Check dims
    #     self.assertEqual(result.shape[0], 1)
    #     self.assertEqual(result.shape[1], input.shape[1])

    #     # Check data
    #     for result_row, expected_row in zip(result.data, expected_result.data):
    #         for result_value, expected_value in zip(result_row, expected_row):
    #             self.assertAlmostEqual(result_value.data, expected_value.data)

    # def test_tensor_mse(self):
    #     input = tensor([[2, 2, 3], 
    #                                   [5, 1, 0], 
    #                                   [1, 2, 3]])

    #     target = tensor([[0, 1, 0], 
    #                                    [1, 0, 0], 
    #                                    [1, 0, 0]])
        
    #     result = input.MSE(target)
    #     expected_result = tensor.from_1d_array([4.666666666, 5.666666666, 4.33333333])
        
    #     # Check dims
    #     self.assertEqual(result.shape[0], 1)
    #     self.assertEqual(result.shape[1], input.shape[1])

    #     # Check data
    #     for result_row, expected_row in zip(result.data, expected_result.data):
    #         for result_value, expected_value in zip(result_row, expected_row):
    #             self.assertAlmostEqual(result_value.data, expected_value.data)

if __name__ == "__main__":
    unittest.main()
class TestTensorGenerationFuncs(unittest.TestCase):
    def test_tensor_full(self):
        a = Tensor.full((3, 2), 1.3)

        # Check dims
        self.assertEqual(a.shape[0], 3)
        self.assertEqual(a.shape[1], 2)

        # Check data
        self.assertTrue(all((a == 1.3)._np.flatten().tolist()))