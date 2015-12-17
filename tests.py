import unittest2 as unittest

class MyTest(unittest.TestCase):
    def test_basic_one_plus_one(self):
        self.assertEqual(1+1,2)


if __name__ == '__main__':
    unittest.main()