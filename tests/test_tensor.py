import unittest
from minitorch.tensor import Tensor
from minitorch.storage import Storage

class TestTensor(unittest.TestCase):
    def test_init(self):
        # Test initialization with data and dtype
        data = [1, 2, 3]
        dtype = Tensor.Dtype.Int
        tensor = Tensor(data, dtype=dtype)
        self.assertTrue(tensor._storage.eq(Storage(data)))
        self.assertEqual(tensor.dtype, dtype)
        self.assertFalse(tensor.requires_grad)
        self.assertIsNone(tensor.grad)
        self.assertIsNone(tensor._ctx)

        # Test initialization with Storage object
        storage = Storage(data)
        tensor = Tensor(storage)
        self.assertTrue(tensor._storage.eq(storage))
        self.assertFalse(tensor.requires_grad)
        self.assertIsNone(tensor.grad)
        self.assertIsNone(tensor._ctx)

        # Test initialization with data, dtype, and requires_grad
        requires_grad = True
        tensor = Tensor(data, dtype=dtype, requires_grad=requires_grad)
        self.assertTrue(tensor._storage.eq(Storage(data)))
        self.assertEqual(tensor.dtype, dtype)
        self.assertTrue(tensor.requires_grad)
        self.assertIsNone(tensor.grad)
        self.assertIsNone(tensor._ctx)

    def test_shape(self):
        # Test shape property
        data = [[1, 2, 3], [4, 5, 6]]
        tensor = Tensor(data)
        self.assertEqual(tensor.shape, (2, 3))

    def test_ndim(self):
        # Test ndim property
        data = [[1, 2, 3], [4, 5, 6]]
        tensor = Tensor(data)
        self.assertEqual(tensor.ndim, 2)

    def test_T(self):
        # Test T property
        data = [[1, 2, 3], [4, 5, 6]]
        tensor = Tensor(data)
        transposed_tensor = tensor.T
        expected_data = [[1, 4], [2, 5], [3, 6]]
        self.assertTrue(transposed_tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(transposed_tensor.shape, (3, 2))

    def test_full(self):
        # Test full static method
        shape = (2, 3)
        value = 5
        tensor = Tensor.full(shape, value)
        expected_data = [[5, 5, 5], [5, 5, 5]]
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, shape)

    def test_full_like(self):
        # Test full_like static method
        shape = (2, 3)
        value = 5
        other = Tensor.ones(shape)
        tensor = Tensor.full_like(other, value)
        expected_data = [[5, 5, 5], [5, 5, 5]]
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, shape)

    def test_zeros(self):
        # Test zeros static method
        shape = (2, 3)
        tensor = Tensor.zeros(shape)
        expected_data = [[0, 0, 0], [0, 0, 0]]
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, shape)

    def test_ones(self):
        # Test ones static method
        shape = (2, 3)
        tensor = Tensor.ones(shape)
        expected_data = [[1, 1, 1], [1, 1, 1]]
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, shape)

    def test_arange(self):
        # Test arange static method
        start = 0
        end = 5
        tensor = Tensor.arange(start, end)
        expected_data = [0, 1, 2, 3, 4]
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, (5,))

    def test_one_hot(self):
        # Test one_hot static method
        n_classes = 5
        hot_class = 2
        tensor = Tensor.one_hot(n_classes, hot_class)
        expected_data = [0, 0, 1, 0, 0]
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, (5,))

    def test_randn(self):
        # Test randn static method
        shape = (2, 3)
        tensor = Tensor.randn(shape)
        self.assertEqual(tensor.shape, shape)

    def test_uniform(self):
        # Test uniform static method
        shape = (2, 3)
        low = 0
        high = 1
        tensor = Tensor.uniform(shape, low, high)
        self.assertEqual(tensor.shape, shape)

    def test_masked_fill(self):
        # Test masked_fill function
        data = [1, 2, 3, 4, 5]
        mask = Tensor([True, False, True, False, True])
        value = 10
        tensor = Tensor(data)
        filled_tensor = tensor.masked_fill(mask, value)
        expected_data = [10, 2, 10, 4, 10]
        self.assertTrue(filled_tensor._storage.eq(Storage(expected_data)))

    def test_replace(self):
        # Test replacing a single value
        data = [3, 2, 3, 4, 5]
        target = 3
        new_value = 10
        tensor = Tensor(data)
        replaced_tensor = tensor.replace(target, new_value)
        expected_data = [10, 2, 10, 4, 5]
        self.assertTrue(replaced_tensor._storage.eq(Storage(expected_data)))

    def test__tri(self):
        # Test _tri method with positive offset
        row = 4
        col = 4
        offset = 1
        tensor = Tensor._tri(row, col, offset=offset)
        expected_data = [[1, 0, 0, 0],
                         [1, 1, 0, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 1]]
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, (row, col))

        # Test _tri method with negative offset
        offset = -1
        tensor = Tensor._tri(row, col, offset=offset)
        expected_data = [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [1, 1, 0, 0]]
        
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, (row, col))

        # Test _tri method with zero offset
        offset = 0
        tensor = Tensor._tri(row, col, offset=offset)
        expected_data = [[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [1, 1, 0, 0],
                         [1, 1, 1, 0]]
        
        self.assertTrue(tensor._storage.eq(Storage(expected_data)))
        self.assertEqual(tensor.shape, (row, col))

    def test_add_scalar(self):
        # Test addition of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        result = tensor + 5
        expected_data = [6, 7, 8]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_add_tensor(self):
        # Test addition of two tensors
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([4, 5, 6])
        result = tensor1 + tensor2
        expected_data = [5, 7, 9]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_add_scalar(self):
        # Test reverse addition of scalar with a tensor
        tensor = Tensor([1, 2, 3])
        result = 5 + tensor
        expected_data = [6, 7, 8]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_add_tensor(self):
        # Test reverse addition of tensor with another tensor
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([4, 5, 6])
        result = tensor2 + tensor1
        expected_data = [5, 7, 9]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_iadd_scalar(self):
        # Test in-place addition of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        tensor += 5
        expected_data = [6, 7, 8]
        self.assertTrue(tensor._storage.eq(Tensor(expected_data)._storage))

    def test_iadd_tensor(self):
        # Test in-place addition of two tensors
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([4, 5, 6])
        tensor1 += tensor2
        expected_data = [5, 7, 9]
        self.assertTrue(tensor1._storage.eq(Tensor(expected_data)._storage))

    def test_sub_scalar(self):
        # Test subtraction of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        result = tensor - 1
        expected_data = [0, 1, 2]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_sub_tensor(self):
        # Test subtraction of two tensors
        tensor1 = Tensor([4, 5, 6])
        tensor2 = Tensor([1, 2, 3])
        result = tensor1 - tensor2
        expected_data = [3, 3, 3]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_sub_scalar(self):
        # Test reverse subtraction of scalar with a tensor
        tensor = Tensor([1, 2, 3])
        result = 5 - tensor
        expected_data = [4, 3, 2]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_sub_tensor(self):
        # Test reverse subtraction of tensor with another tensor
        tensor1 = Tensor([4, 5, 6])
        tensor2 = Tensor([1, 2, 3])
        result = tensor2 - tensor1
        expected_data = [-3, -3, -3]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_isub_scalar(self):
        # Test in-place subtraction of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        tensor -= 1
        expected_data = [0, 1, 2]
        self.assertTrue(tensor._storage.eq(Tensor(expected_data)._storage))

    def test_isub_tensor(self):
        # Test in-place subtraction of two tensors
        tensor1 = Tensor([4, 5, 6])
        tensor2 = Tensor([1, 2, 3])
        tensor1 -= tensor2
        expected_data = [3, 3, 3]
        self.assertTrue(tensor1._storage.eq(Tensor(expected_data)._storage))

    def test_scalar_multiplication(self):
        # Test scalar multiplication
        tensor = Tensor([1, 2, 3])
        result = tensor * 2
        expected_data = [2, 4, 6]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_elementwise_multiplication(self):
        # Test elementwise multiplication of two tensors
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([4, 5, 6])
        result = tensor1 * tensor2
        expected_data = [4, 10, 18]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_scalar_multiplication(self):
        # Test reverse scalar multiplication
        tensor = Tensor([1, 2, 3])
        result = 2 * tensor
        expected_data = [2, 4, 6]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_elementwise_multiplication(self):
        # Test reverse elementwise multiplication of two tensors
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([4, 5, 6])
        result = tensor2 * tensor1
        expected_data = [4, 10, 18]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_imul_scalar(self):
        # Test in-place multiplication of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        tensor *= 2
        expected_data = [2, 4, 6]
        self.assertTrue(tensor._storage.eq(Tensor(expected_data)._storage))

    def test_imul_tensor(self):
        # Test in-place multiplication of tensor with another tensor
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([4, 5, 6])
        tensor1 *= tensor2
        expected_data = [4, 10, 18]
        self.assertTrue(tensor1._storage.eq(Tensor(expected_data)._storage))

    def test_truediv_scalar(self):
        # Test division of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        result = tensor / 2
        expected_data = [0.5, 1.0, 1.5]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_truediv_tensor(self):
        # Test division of two tensors
        tensor1 = Tensor([4, 5, 6])
        tensor2 = Tensor([2, 2, 2])
        result = tensor1 / tensor2
        expected_data = [2.0, 2.5, 3.0]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_truediv_scalar(self):
        # Test reverse division of scalar with a tensor
        tensor = Tensor([2, 4, 6])
        result = 10 / tensor
        expected_data = [5.0, 2.5, 1.6666666666666667]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_truediv_tensor(self):
        # Test reverse division of tensor with another tensor
        tensor1 = Tensor([10, 20, 30])
        tensor2 = Tensor([2, 4, 6])
        result = tensor2 / tensor1
        expected_data = [0.2, 0.2, 0.2]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_itruediv_scalar(self):
        # Test in-place division of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        tensor /= 2
        expected_data = [0.5, 1.0, 1.5]
        self.assertTrue(tensor._storage.eq(Tensor(expected_data)._storage))

    def test_itruediv_tensor(self):
        # Test in-place division of tensor with another tensor
        tensor1 = Tensor([4, 6, 8])
        tensor2 = Tensor([2, 3, 4])
        tensor1 /= tensor2
        expected_data = [2.0, 2.0, 2.0]
        self.assertTrue(tensor1._storage.eq(Tensor(expected_data)._storage))

    def test_scalar_power(self):
        # Test power of tensor with a scalar
        tensor = Tensor([1, 2, 3])
        result = tensor ** 2
        expected_data = [1, 4, 9]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_tensor_power(self):
        # Test power of two tensors
        tensor1 = Tensor([2, 3, 4])
        tensor2 = Tensor([2, 2, 2])
        result = tensor1 ** tensor2
        expected_data = [4, 9, 16]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_scalar_power(self):
        # Test reverse power of scalar with a tensor
        tensor = Tensor([2, 3, 4])
        result = 2 ** tensor
        expected_data = [4, 8, 16]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_reverse_tensor_power(self):
        # Test reverse power of tensor with another tensor
        tensor1 = Tensor([2, 3, 4])
        tensor2 = Tensor([2, 2, 2])
        result = tensor2 ** tensor1
        expected_data = [4, 8, 16]
        self.assertTrue(result._storage.eq(Tensor(expected_data)._storage))

    def test_ipow_scalar(self):
        # Test in-place exponentiation with a scalar
        tensor = Tensor([2, 3, 4])
        tensor **= 2
        expected_data = [4, 9, 16]
        self.assertTrue(tensor._storage.eq(Tensor(expected_data)._storage))

    def test_ipow_tensor(self):
        # Test in-place exponentiation with another tensor
        tensor1 = Tensor([2, 3, 4])
        tensor2 = Tensor([2, 2, 2])
        tensor1 **= tensor2
        expected_data = [4, 9, 16]
        self.assertTrue(tensor1._storage.eq(Tensor(expected_data)._storage))

    def test_matmul(self):
        # Test matrix multiplication of two tensors
        tensor1 = Tensor([[1, 2], [3, 4]])
        tensor2 = Tensor([[5, 6], [7, 8]])
        result = tensor1 @ tensor2
        expected_data = [[19, 22], [43, 50]]
        self.assertTrue(result._storage.eq(Storage(expected_data)))

    def test_imatmul(self):
        # Test matrix multiplication with in-place assignment
        tensor1 = Tensor([[1, 2], [3, 4]])
        tensor2 = Tensor([[5, 6], [7, 8]])
        tensor1 @= tensor2
        expected_data = [[19, 22], [43, 50]]
        self.assertTrue(tensor1._storage.eq(Tensor(expected_data)._storage))

        # Test matrix multiplication with in-place assignment and broadcasting
        tensor1 = Tensor([1, 2])
        tensor2 = Tensor([[3, 4], [5, 6]])
        tensor1 @= tensor2
        expected_data = [13, 16]
        self.assertTrue(tensor1._storage.eq(Tensor(expected_data)._storage))

    def test_exp(self):
        # Test case 1: exp of a positive number
        x = Tensor(2)
        result = x.exp()
        expected_result = Tensor(7.3890560989306495)
        self.assertTrue(result._storage.eq(expected_result._storage))

        # Test case 2: exp of a negative number
        x = Tensor(-2)
        result = x.exp()
        expected_result = Tensor(0.1353352832366127)
        self.assertTrue(result._storage.eq(expected_result._storage))

        # Test case 3: exp of zero
        x = Tensor(0)
        result = x.exp()
        expected_result = Tensor(1)
        self.assertTrue(result._storage.eq(expected_result._storage))

    def test_sqrt(self):
        # Test case 1: sqrt of a positive number
        x = Tensor(4)
        result = x.sqrt()
        expected_result = Tensor(2)
        self.assertTrue(result._storage.eq(expected_result._storage))

        # Test case 2: sqrt of a negative number
        # x = Tensor(-4)
        # result = x.sqrt()
        # expected_result = Tensor(float('nan'))
        # self.assertFalse(result._storage.eq(result._storage))  # Check for NaN

        # Test case 3: sqrt of zero
        x = Tensor(0)
        result = x.sqrt()
        expected_result = Tensor(0)
        self.assertTrue(result._storage.eq(expected_result._storage))    
    
    def test_where(self):
        # Test case 1: Condition is True
        condition = Tensor([True, True, True])
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])
        result = condition.where(x, y)
        expected_result = Tensor([1, 2, 3])
        self.assertTrue(result._storage.eq(expected_result._storage))

        # Test case 2: Condition is False
        condition = Tensor([False, False, False])
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])
        result = condition.where(x, y)
        expected_result = Tensor([4, 5, 6])
        self.assertTrue(result._storage.eq(expected_result._storage))

        # Test case 3: Condition is a Tensor with different shape
        condition = Tensor([True, False])
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])
        try:
            result = condition.where(x, y)
            expected_result = Tensor([1, 5, 6])
            self.assertTrue(result._storage.eq(expected_result._storage))
        except ValueError as e:
            self.assertEqual(str(e), \
                f"operands could not be broadcast together with remapped shapes [original->remapped]: {condition.shape}  and requested shape {x.shape}")

    def test_sigmoid(self):
        # Test sigmoid function
        x = Tensor([-10, -5, 0, 5, 10], dtype=Tensor.Dtype.Float)
        result = x.sigmoid()
        expected_result = Tensor([0.00004539, 0.00669285, 0.5, 0.99330715, 0.99995460])
        self.assertTrue(result._storage.all_close(expected_result._storage))

    def test_tanh(self):
        # Test tanh function
        x = Tensor([-10, -5, 0, 5, 10], dtype=Tensor.Dtype.Float)
        result = x.tanh()
        expected_result = Tensor([-1.00000000, -0.99990920, 0.00000000, 0.99990920, 1.00000000])
        self.assertTrue(result._storage.all_close(expected_result._storage))

    def test_relu(self):
        # Test relu function
        x = Tensor([-10, -5, 0, 5, 10], dtype=Tensor.Dtype.Float)
        result = x.relu()
        expected_result = Tensor([0, 0, 0, 5, 10])
        self.assertTrue(result._storage.all_close(expected_result._storage))

    def test_softmax(self):
        # Test softmax function
        x = Tensor([1, 2, 3, 4, 5], dtype=Tensor.Dtype.Float)
        result = x.softmax()
        expected_result = Tensor([0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865])
        self.assertTrue(result._storage.all_close(expected_result._storage))

    def test_MSE(self):
        # Test case 1: MSE of two tensors
        tensor1 = Tensor([1, 2, 3])
        tensor2 = Tensor([4, 5, 6])
        result = tensor1.MSE(tensor2)
        expected_result = Tensor(9)
        self.assertTrue(result._storage.all_close(expected_result._storage))

        # Test case 2: MSE of two tensors with axis
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]])
        tensor2 = Tensor([[4, 5, 6], [1, 2, 3]])
        result = tensor1.MSE(tensor2, axis=1)
        expected_result = Tensor([9, 9])
        self.assertTrue(result._storage.all_close(expected_result._storage))

    def test_cross_entropy(self):
        # Test case 1: Cross entropy of two tensors
        tensor1 = Tensor([0.2, 0.3, 0.5])
        tensor2 = Tensor([0.3, 0.3, 0.4])
        result = tensor1.cross_entropy(tensor2)
        expected_result = Tensor(1.12128209)
        self.assertTrue(result._storage.all_close(expected_result._storage))

        # Test case 2: Cross entropy of two tensors with axis
        tensor1 = Tensor([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
        tensor2 = Tensor([[0.3, 0.3, 0.4], [0.3, 0.4, 0.3]])
        result = tensor1.cross_entropy(tensor2, axis=1)
        expected_result = Tensor([1.12128209, 1.11766818])
        self.assertTrue(result._storage.all_close(expected_result._storage))

    def test_broadcasted(self):
        # Test case 1: Same shape tensors
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])
        result_x, result_y = x._broadcasted(y)
        self.assertTrue(result_x._storage.eq(x._storage))
        self.assertTrue(result_y._storage.eq(y._storage))

        # Test case 2: Different shape tensors
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = Tensor([7, 8, 9])
        result_x, result_y = x._broadcasted(y)
        expected_x = Tensor([[1, 2, 3], [4, 5, 6]])
        expected_y = Tensor([[7, 8, 9], [7, 8, 9]])
        self.assertTrue(result_x._storage.eq(expected_x._storage))
        self.assertTrue(result_y._storage.eq(expected_y._storage))

        # Test case 3: Broadcasting with scalar tensor
        x = Tensor([1, 2, 3])
        y = Tensor(4)
        result_x, result_y = x._broadcasted(y)
        expected_x = Tensor([1, 2, 3])
        expected_y = Tensor([4, 4, 4])
        self.assertTrue(result_x._storage.eq(expected_x._storage))
        self.assertTrue(result_y._storage.eq(expected_y._storage))
    
    def test_getitem_single_index(self):
        # Test getting a single element from a 1D tensor
        tensor = Tensor([1, 2, 3, 4, 5])
        self.assertEqual(tensor[2], 3)

    def test_getitem_slice(self):
        # Test getting a slice from a 1D tensor
        tensor = Tensor([1, 2, 3, 4, 5])
        self.assertEqual(tensor[1:4], Tensor([2, 3, 4]))

    def test_getitem_ellipsis(self):
        # Test using ellipsis to select all elements
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(tensor[..., 1], Tensor([2, 5, 8]))

    def test_getitem_negative_indices(self):
        # Test using negative indices to select elements from the end
        tensor = Tensor([1, 2, 3, 4, 5])
        self.assertEqual(tensor[-3:-1], Tensor([3, 4]))

    def test_getitem_strides(self):
        # Test using strides to select elements with a step
        tensor = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(tensor[1:9:2], Tensor([2, 4, 6, 8]))

    def test_getitem_tensor(self):
        # Test using strides to select elements with a step
        tensor = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        idx = Tensor([0, 4, 7, 9])
        self.assertTrue(tensor[idx]._storage.eq(Tensor([1, 5, 8, 10])._storage))

if __name__ == '__main__':
    unittest.main()