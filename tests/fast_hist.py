import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from fast_intensity.fast_hist import fast_hist

class TestFastHistFunction(unittest.TestCase):
    def test_returns_correct_density_values_for_uniform_data(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        edges = np.array([0, 5.5, 11])
        result = fast_hist(x, edges)
        self.assertEqual(result[0], result[1])

    def test_returns_correct_density_values_for_one_val(self):
        x = np.array([5])
        edges = np.array([4, 6])
        self.assertEqual(fast_hist(x, edges), 0.5)

    def test_identical_result_for_list_and_array(self):
        x_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        edges_list = [0, 4, 6, 10]
        x = np.array(x_list)
        edges = np.array(edges_list)
        npt.assert_array_equal(fast_hist(x_list, edges_list),
                               fast_hist(x, edges))

    def test_raises_exception_when_edges_unsorted(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        edges = np.array([10, 4, 6, 0])
        self.assertRaises(Exception, fast_hist, (x, edges))

    def test_raises_exception_when_values_unsorted(self):
        x = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        edges = np.array([0, 4, 6, 10])
        self.assertRaises(Exception, fast_hist, (x, edges))

if __name__ == '__main__':
    unittest.main()
