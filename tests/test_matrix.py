import unittest
import math

from minitorch.matrix import Matrix

class TestMatrixGenerationFuncs(unittest.TestCase):
    def test_matrix_fill(self):
        a = Matrix.fill(3, 2, 1.3)

        # Check dims
        self.assertEqual(a.shape.row, 3)
        self.assertEqual(a.shape.col, 2)

        # Check data
        for row in a.data:
            for value in row:
                self.assertEqual(value.data, 1.3)

    def test_matrix_zeros(self):
        a = Matrix.zeros(2, 4)

        # Check dims
        self.assertEqual(a.shape.row, 2)
        self.assertEqual(a.shape.col, 4)

        # Check data
        for row in a.data:
            for value in row:
                self.assertEqual(value.data, 0.0)

    def test_matrix_randn(self):
        a = Matrix.randn(5, 1)

        # Check dims
        self.assertEqual(a.shape.row, 5)
        self.assertEqual(a.shape.col, 1)

        # Impossible to check random data 
    
    def test_matrix_masked_fill(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])

        new = Matrix.masked_fill(old, old == 3.2, 1.7)

        # Check dims
        self.assertEqual(new.shape.row, 3)
        self.assertEqual(new.shape.col, 3)
        
        # Check data
        for old_row, new_row in zip(old.data, new.data):
            for old_value, new_value in zip(old_row, new_row):
                if old_value.data == 3.2:
                    self.assertEqual(new_value.data, 1.7)

    def test_matrix_replace(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])

        new = Matrix.replace(old, 3.2, 1.7)

        # Check dims
        self.assertEqual(new.shape.row, 3)
        self.assertEqual(new.shape.col, 3)

        # Check data
        for old_row, new_row in zip(old.data, new.data):
            for old_value, new_value in zip(old_row, new_row):
                if old_value.data == 3.2:
                    self.assertEqual(new_value.data, 1.7)

    def test_matrix_tril_main_diagonal(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])

        new = Matrix.tril(old)
        expected_new = Matrix.from_2d_array([[ 0.7,  0.0, 0.0], 
                                             [ 3.2, -3.9, 0.0], 
                                             [-1.5,  3.2, 3.2]])

        # Check dims
        self.assertEqual(new.shape.row, 3)
        self.assertEqual(new.shape.col, 3)
        
        # Check data
        for new_row, expected_new_row in zip(new.data, expected_new.data):
            for new_value, expected_new_value in zip(new_row, expected_new_row):
                self.assertAlmostEqual(new_value.data, expected_new_value.data)

    def test_matrix_tril_anti_diagonal(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])

        new = Matrix.tril(old, Matrix.Diagonal.ANTI)
        expected_new = Matrix.from_2d_array([[0.7,  3.2, 1.1], 
                                             [0.0, -3.9, 0.2], 
                                             [0.0,  0.0, 3.2]])
        # Check dims
        self.assertEqual(new.shape.row, 3)
        self.assertEqual(new.shape.col, 3)
        
        # Check data
        for new_row, expected_new_row in zip(new.data, expected_new.data):
            for new_value, expected_new_value in zip(new_row, expected_new_row):
                self.assertAlmostEqual(new_value.data, expected_new_value.data)

class TestBinaryOps(unittest.TestCase):
    def test_matrix_is_equal_to_matrix(self):
        a = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3], 
                                  [2.1,  1.4,  0.8]])
        
        b = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3], 
                                  [2.1,  1.4,  0.8]])
        
        self.assertEqual(a == b, True)
        
    def test_matrix_is_elementwise_equal_to_float(self):
        a = Matrix.from_2d_array([[ 0.7,  3.2, 1.1],  
                                  [-1.5,  3.2, 3.2]])
        
        target = 3.2

        result = a == target
        expected_result = [[False, True , False],
                           [False, True , True ],]
        
        # Check result
        for res_row, exp_res_row in zip(result, expected_result):
            for res_val, exp_res_val in zip(res_row, exp_res_row):
                self.assertEqual(res_val, exp_res_val) 

    def test_add_two_matrix_objects(self):
        a = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3], 
                                  [2.1,  1.4,  0.8]])
        
        b = Matrix.from_2d_array([[ 0.8,  1.4, 2.1], 
                                  [ 2.3, -0.5, 1.2], 
                                  [-1.8,  2.4, 3.0]])
        
        result = a + b

        # Check result
        expected_result = Matrix.from_2d_array([[ 3.8,  3.8, 0.3], 
                                       [ 3.5, -1.0, 3.5], 
                                       [ 0.3,  3.8, 3.8]])

        rows, cols = result.shape.row, result.shape.col
        for row in range(rows):
            for col in range(cols):
                self.assertAlmostEqual(result[row][col].data, expected_result[row][col].data)

    def test_multiply_matrix_and_float(self):
        a = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3], 
                                  [2.1,  1.4,  0.8]])
        
        b = 1.7

        result = a * b

        # Check result
        expected_result = Matrix.from_2d_array([[3.0 * 1.7,  2.4 * 1.7, -1.8 * 1.7], 
                                                [1.2 * 1.7, -0.5 * 1.7,  2.3 * 1.7], 
                                                [2.1 * 1.7,  1.4 * 1.7,  0.8 * 1.7]])

        rows, cols = result.shape.row, result.shape.col
        for row in range(rows):
            for col in range(cols):
                self.assertAlmostEqual(result[row][col].data, expected_result[row][col].data)

    def test_multiply_float_and_matrix(self):
        a = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3], 
                                  [2.1,  1.4,  0.8]])
        
        b = 1.7

        result = b * a

        # Check result
        expected_result = Matrix.from_2d_array([[3.0 * 1.7,  2.4 * 1.7, -1.8 * 1.7], 
                                                [1.2 * 1.7, -0.5 * 1.7,  2.3 * 1.7], 
                                                [2.1 * 1.7,  1.4 * 1.7,  0.8 * 1.7]])

        rows, cols = result.shape.row, result.shape.col
        for row in range(rows):
            for col in range(cols):
                self.assertAlmostEqual(result[row][col].data, expected_result[row][col].data)

    def test_divide_matrix_and_float(self):
        a = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3], 
                                  [2.1,  1.4,  0.8]])
        
        b = 1.7

        result = a / b

        # Check result
        expected_result = Matrix.from_2d_array([[3.0 / 1.7,  2.4 / 1.7, -1.8 / 1.7], 
                                                [1.2 / 1.7, -0.5 / 1.7,  2.3 / 1.7], 
                                                [2.1 / 1.7,  1.4 / 1.7,  0.8 / 1.7]])

        rows, cols = result.shape.row, result.shape.col
        for row in range(rows):
            for col in range(cols):
                self.assertAlmostEqual(result[row][col].data, expected_result[row][col].data)

    def test_matrix_matmul_product(self):
        a = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3], 
                                  [2.1,  1.4,  0.8]])
        
        b = Matrix.from_2d_array([[0.8],  [1.4], [2.1]])
        
        result = a.matmul(b)
        
        # Check shape
        self.assertEqual(a.shape.row, result.shape.row)
        self.assertEqual(b.shape.col, result.shape.col)

        # Check result
        expected_result = Matrix.from_2d_array([[1.98], [5.09], [5.32]])

        rows, cols = result.shape.row, result.shape.col
        for row in range(rows):
            for col in range(cols):
                self.assertAlmostEqual(result[row][col].data, expected_result[row][col].data)

class TestUnaryOps(unittest.TestCase):
    def test_matrix_transpose(self):
        a = Matrix.from_2d_array([[3.0,  2.4, -1.8], 
                                  [1.2, -0.5,  2.3]])
        
        result = a.T()

        # Check shape
        self.assertEqual(a.shape.row, result.shape.col)
        self.assertEqual(a.shape.col, result.shape.row)

        # Check result
        expected_result = Matrix.from_2d_array([[ 3.0,  1.2],
                                                [ 2.4, -0.5],
                                                [-1.8,  2.3]])

        rows, cols = result.shape.row, result.shape.col
        for row in range(rows):
            for col in range(cols):
                self.assertAlmostEqual(result[row][col].data, expected_result[row][col].data)

    def test_matrix_flatten(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])
        
        new = old.flatten()
        expected_new = Matrix.from_1d_array([0.7,  3.2, 1.1, 3.2, -3.9, 0.2, -1.5,  3.2, 3.2])

        # Check dims
        self.assertEqual(new.shape.row, 1)
        self.assertEqual(new.shape.col, (old.shape.row * old.shape.col))
        
        # Check data
        for new_row, expected_new_row in zip(new.data, expected_new.data):
            for new_value, expected_new_value in zip(new_row, expected_new_row):
                self.assertAlmostEqual(new_value.data, expected_new_value.data)

    def test_matrix_sum_dim_not_specified(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])
        
        result = old.sum()
        expected_result = sum(*old.flatten().data).data

        # Check dims
        self.assertEqual(result.shape.row, 1)
        self.assertEqual(result.shape.col, 1)
        
        # Check data
        self.assertEqual(result.item().data, expected_result)

    def test_matrix_sum_dim_0(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])
        
        result = old.sum(dim=0)
        expected_result = Matrix.from_1d_array([sum(row).data for row in old.T().data])

        # Check dims
        self.assertEqual(result.shape.row, 3)
        self.assertEqual(result.shape.col, 1)
        
        # Check data
        for result_row, expected_result_row in zip(result.data, expected_result.data):
            for result_value, expected_result_value in zip(result_row, expected_result_row):
                self.assertAlmostEqual(result_value.data, expected_result_value.data)

    def test_matrix_sum_dim_1(self):
        old = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                    [ 3.2, -3.9, 0.2], 
                                    [-1.5,  3.2, 3.2]])
        
        result = old.sum(dim=1)
        expected_result = Matrix.from_1d_array([sum(row).data for row in old.data])

        # Check dims
        self.assertEqual(result.shape.row, 3)
        self.assertEqual(result.shape.col, 1)
        
        # Check data
        for result_row, expected_result_row in zip(result.data, expected_result.data):
            for result_value, expected_result_value in zip(result_row, expected_result_row):
                self.assertAlmostEqual(result_value.data, expected_result_value.data)

    def test_matrix_exp(self):
        input = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                      [ 3.2, -3.9, 0.2], 
                                      [-1.5,  3.2, 3.2]])
        
        result = input.exp()

        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for input_row, result_row in zip(input.data, result.data):
            for input_value, result_value in zip(input_row, result_row):
                self.assertAlmostEqual(input_value.exp().data, result_value.data)

    def test_matrix_log_base_2(self):
        input = Matrix.from_2d_array([[0.7, 3.2, 1.1], 
                                      [3.2, 3.9, 0.2], 
                                      [1.5, 3.2, 3.2]])
        
        result = input.log(2)

        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for input_row, result_row in zip(input.data, result.data):
            for input_value, result_value in zip(input_row, result_row):
                self.assertAlmostEqual(input_value.log(2).data, result_value.data)

    def test_matrix_log_base_e(self):
        input = Matrix.from_2d_array([[0.7, 3.2, 1.1], 
                                      [3.2, 3.9, 0.2], 
                                      [1.5, 3.2, 3.2]])
        
        result = input.log(math.e)

        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for input_row, result_row in zip(input.data, result.data):
            for input_value, result_value in zip(input_row, result_row):
                self.assertAlmostEqual(input_value.log(math.e).data, result_value.data)

    def test_matrix_sigmoid(self):
        input = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                      [ 3.2, -3.9, 0.2], 
                                      [-1.5,  3.2, 3.2]])
        
        result = input.sigmoid()

        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for input_row, result_row in zip(input.data, result.data):
            for input_value, result_value in zip(input_row, result_row):
                self.assertAlmostEqual(input_value.sigmoid().data, result_value.data)

    def test_matrix_relu(self):
        input = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                      [ 3.2, -3.9, 0.2], 
                                      [-1.5,  3.2, 3.2]])
        
        result = input.relu()

        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for input_row, result_row in zip(input.data, result.data):
            for input_value, result_value in zip(input_row, result_row):
                self.assertAlmostEqual(input_value.relu().data, result_value.data)

    def test_matrix_tanh(self):
        input = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                      [ 3.2, -3.9, 0.2], 
                                      [-1.5,  3.2, 3.2]])
        
        result = input.tanh()

        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for input_row, result_row in zip(input.data, result.data):
            for input_value, result_value in zip(input_row, result_row):
                self.assertAlmostEqual(input_value.tanh().data, result_value.data)

    def test_matrix_softmax_dim_0(self):
        input = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                      [ 3.2, -3.9, 0.2], 
                                      [-1.5,  3.2, 3.2]])
        
        result = input.softmax(dim=0)
        expected_result = Matrix.from_2d_array([[0.0752258819, 0.4997938080, 0.104463303],
                                                [0.9164388527, 0.0004123823, 0.0424716097],
                                                [0.0083352653, 0.4997938088, 0.8530650866]])
        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for result_row, expected_row in zip(result.data, expected_result.data):
            for result_value, expected_value in zip(result_row, expected_row):
                self.assertAlmostEqual(result_value.data, expected_value.data)

    def test_matrix_softmax_dim_1(self):
        input = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                      [ 3.2, -3.9, 0.2], 
                                      [-1.5,  3.2, 3.2]])
        
        result = input.softmax(dim=1)
        expected_result = Matrix.from_2d_array([[0.06814626, 0.83019145, 0.101662280],
                                                [0.951826016, 7.8536e-04, 0.0473886269],
                                                [4.5271e-03, 0.497736474, 0.4977364744]])
        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for result_row, expected_row in zip(result.data, expected_result.data):
            for result_value, expected_value in zip(result_row, expected_row):
                self.assertAlmostEqual(result_value.data, expected_value.data)

    def test_matrix_softmax_dim_1(self):
        input = Matrix.from_2d_array([[ 0.7,  3.2, 1.1], 
                                      [ 3.2, -3.9, 0.2], 
                                      [-1.5,  3.2, 3.2]])
        
        result = input.softmax(dim=1)
        expected_result = Matrix.from_2d_array([[0.06814626, 0.83019145, 0.101662280],
                                                [0.951826016, 7.8536e-04, 0.0473886269],
                                                [4.5271e-03, 0.497736474, 0.4977364744]])
        
        # Check dims
        self.assertEqual(result.shape.row, input.shape.row)
        self.assertEqual(result.shape.col, input.shape.col)
        
        # Check data
        for result_row, expected_row in zip(result.data, expected_result.data):
            for result_value, expected_value in zip(result_row, expected_row):
                self.assertAlmostEqual(result_value.data, expected_value.data)

    def test_matrix_cross_entropy(self):
        input = Matrix.from_2d_array([[0.7, 0.2, 0.1], 
                                      [0.5, 0.25, 0.25], 
                                      [0.95, 0.01, 0.04]])

        target = Matrix.from_2d_array([[0, 1, 0], 
                                       [1, 0, 0], 
                                       [1, 0, 0]])
        
        result = input.cross_entropy(target)
        expected_result = Matrix.from_1d_array([2.321928094, 1.0000, 0.07400058144])
        
        # Check dims
        self.assertEqual(result.shape.row, 1)
        self.assertEqual(result.shape.col, input.shape.col)

        # Check data
        for result_row, expected_row in zip(result.data, expected_result.data):
            for result_value, expected_value in zip(result_row, expected_row):
                self.assertAlmostEqual(result_value.data, expected_value.data)

    def test_matrix_mse(self):
        input = Matrix.from_2d_array([[2, 2, 3], 
                                      [5, 1, 0], 
                                      [1, 2, 3]])

        target = Matrix.from_2d_array([[0, 1, 0], 
                                       [1, 0, 0], 
                                       [1, 0, 0]])
        
        result = input.MSE(target)
        expected_result = Matrix.from_1d_array([4.666666666, 5.666666666, 4.33333333])
        
        # Check dims
        self.assertEqual(result.shape.row, 1)
        self.assertEqual(result.shape.col, input.shape.col)

        # Check data
        for result_row, expected_row in zip(result.data, expected_result.data):
            for result_value, expected_value in zip(result_row, expected_row):
                self.assertAlmostEqual(result_value.data, expected_value.data)

if __name__ == "__main__":
    unittest.main()
