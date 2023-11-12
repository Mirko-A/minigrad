import unittest

from mymodule import Value

class TestValue(unittest.TestCase):
    def test_add_two_value_objects(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        v2 = Value(b)
        self.assertEqual((v1 + v2).val, a + b)

    def test_add_value_object_and_scalar(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((v1 + b).val, a + b)

    def test_add_scalar_and_value_object(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((b + v1).val, a + b)

    def test_mul_two_value_objects(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        v2 = Value(b)
        self.assertEqual((v1 * v2).val, a * b)

    def test_mul_value_object_and_scalar(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((v1 * b).val, a * b)

    def test_mul_scalar_and_value_object(self):
        a = 3.0
        b = 1.5
        v1 = Value(a)
        self.assertEqual((b * v1).val, a * b)

if __name__ == "__main__":
    unittest.main()
