import unittest

import sys
import os
sys.path.append(os.curdir + "\..")

from minigrad import Matrix

class TestBinaryOps(unittest.TestCase):
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
        
        result = Matrix.matmul(a, b)
        
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

if __name__ == "__main__":
    unittest.main()
