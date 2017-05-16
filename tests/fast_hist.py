# Copyright 2017 Thomas A. Lasko, Jacek Bajor

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
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)
        edges = np.array([0, 5.5, 11], dtype=np.float)
        result = fast_hist(x, edges)
        npt.assert_array_equal(result[0], result[1])

    def test_returns_correct_density_values_for_one_val(self):
        x = np.array([5], dtype=np.float)
        edges = np.array([4, 6], dtype=np.float)
        self.assertEqual(fast_hist(x, edges), np.array([0.5]))

if __name__ == '__main__':
    unittest.main()
