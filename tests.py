import unittest2 as unittest
from environment import Storyline

class MyTest(unittest.TestCase):
    def test_basic_one_plus_one(self):
        self.assertEqual(1+1,2)

class TestStoryline(unittest.TestCase):
    def setUp(self):
        self.equal_option_space = (('A',1), ('B',1), ('C',1))
        self.equal_storyline = Storyline(self.equal_option_space)

    def test_basic_setup(self):
        self.assertEqual(self.equal_storyline.weighted_options,\
            [('A',1), ('B',2), ('C',3)])

    def test_additional_option(self):
        self.equal_storyline.add_option('D',1)
        self.assertEqual(self.equal_storyline.weighted_options,\
            [('A',1), ('B',2), ('C',3), ('D',4)])


class TestRex(unittest.TestCase):
    def setUp(self):
        actions = 


if __name__ == '__main__':
    unittest.main()