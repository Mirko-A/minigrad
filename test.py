import unittest
import math

from minigrad import Value

class TestValue(unittest.TestCase):
    def test_add_two_value_objects(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        v2 = Value(b)
        self.assertEqual((v1 + v2).data, a + b)

    def test_add_value_object_and_scalar(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((v1 + b).data, a + b)

    def test_add_scalar_and_value_object(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((b + v1).data, a + b)

    def test_sub_two_value_objects(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        v2 = Value(b)
        self.assertEqual((v1 - v2).data, a - b)

    def test_sub_value_object_and_scalar(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((v1 - b).data, a - b)

    def test_sub_scalar_and_value_object(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((b - v1).data, b - a)

    def test_neg_value_object(self):
        a = 3.0
        v1 = Value(a)
        self.assertEqual(-v1.data, -a)

    def test_mul_two_value_objects(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        v2 = Value(b)
        self.assertEqual((v1 * v2).data, a * b)

    def test_mul_value_object_and_scalar(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((v1 * b).data, a * b)

    def test_mul_scalar_and_value_object(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((b * v1).data, a * b)

    def test_div_two_value_objects(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        v2 = Value(b)
        self.assertEqual((v1 / v2).data, a / b)

    def test_div_value_object_and_scalar(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((v1 / b).data, a / b)

    def test_div_scalar_and_value_object(self):
        a = 3.0
        v1 = Value(a)
        self.assertEqual((1 / v1).data, 1 / a)

    def test_pow_two_value_objects(self):
        a = 3.0
        b = 2.0
        v1 = Value(a)
        v2 = Value(b)
        self.assertEqual((v1 ** v2).data, a ** b)

    def test_pow_value_object_and_scalar(self):
        a = 3.0
        b = 2.0
        v1 = Value(a)
        self.assertEqual((v1 ** b).data, a ** b)

    def test_pow_scalar_and_value_object(self):
        a = 3.0
        b = 2.0
        v1 = Value(a)
        self.assertEqual((b ** v1).data, b ** a)
    
    def test_exp(self):
        a = 3.0
        v1 = Value(a)
        self.assertEqual((math.e ** v1).data, math.e ** a)
        
    def test_sigmoid(self):
        a = 3.0
        v1 = Value(a)
        self.assertEqual((v1.sigmoid()).data, 1 / (1 + math.e ** -a))

    def test_sigmoid(self):
        a = 3.0
        v1 = Value(a)
        self.assertEqual((v1.sigmoid()).data, 1 / (1 + math.e ** -a))

    def test_relu_below_zero(self):
        a = -3.0
        a_relu = 0
        v1 = Value(a)
        self.assertEqual((v1.relu()).data, a_relu)
    
    def test_relu_above_zero(self):
        a = 3.0
        a_relu = a
        v1 = Value(a)
        self.assertEqual((v1.relu()).data, a_relu)

if __name__ == "__main__":
    unittest.main()
