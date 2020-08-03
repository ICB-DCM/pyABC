import unittest

from pyabc.parameters import Parameter
from pyabc.random_variables import RV, Distribution


class TextRVComposition(unittest.TestCase):
    def setUp(self):
        self.d = Distribution(
            **{"a": RV("randint", low=0, high=3+1),
               "b": Distribution(**{"b1": RV("randint", low=0, high=3+1),
                                    "b2": RV("randint", low=0, high=3+1)})})
        self.d_plus_one = Distribution(
            **{"a": RV("randint", low=1, high=1+1),
               "b": Distribution(**{"b1": RV("randint", low=1, high=1+1),
                                    "b2": RV("randint", low=1, high=1+1)})})
        self.x_one = Parameter({"a": 1,
                                "b": Parameter({"b1": 1, "b2": 1})})

        self.x_zero = Parameter({"a": 0,
                                 "b": Parameter({"b1": 0, "b2": 0})})

        self.x_two = Parameter({"a": 2,
                                "b": Parameter({"b1": 2, "b2": 2})})

    def test_composition(self):
        self.assertEqual(1 / 4 ** 3, self.d.pdf(self.x_one))


class TestRVInitialization(unittest.TestCase):
    def test_no_kwargs(self):
        a = RV.from_dictionary({"type": "uniform", "args": [0, 0]})
        self.assertEqual(0, a.rvs())


if __name__ == "__main__":
    unittest.main()
