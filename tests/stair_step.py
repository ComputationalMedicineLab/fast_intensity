import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from fast_intensity.stair_step import stair_step

class TestStairStepFunction(unittest.TestCase):
    def test_returns_correct_interpolation_for_increasing_vals(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float)
        y = np.array([1, 2, 3, 4, 5], dtype=np.float)
        x_new = np.array([2.5, 4.5], dtype=np.float)
        y_new = np.zeros(len(x_new), dtype=np.float)
        expected_y = np.array([2, 4], dtype=np.float)
        npt.assert_array_equal(stair_step(x, y, x_new, y_new), expected_y)

    def test_returns_correct_interpolation_for_decreasing_vals(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float)
        y = np.array([5, 4, 3, 2, 1], dtype=np.float)
        x_new = np.array([2.5, 4.5], dtype=np.float)
        y_new = np.zeros(len(x_new), dtype=np.float)
        expected_y = np.array([4, 2], dtype=np.float)
        npt.assert_array_equal(stair_step(x, y, x_new, y_new), expected_y)

    def test_returns_correct_value_for_last_known_value(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float)
        y = np.array([5, 4, 3, 2, 1], dtype=np.float)
        x_new = np.array([5, 6], dtype=np.float)
        y_new = np.zeros(len(x_new), dtype=np.float)
        expected_y = np.array([1, 1], dtype=np.float)
        npt.assert_array_equal(stair_step(x, y, x_new, y_new), expected_y)

    def test_returns_zero_for_left_out_of_bounds(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float)
        y = np.array([5, 4, 3, 2, 1], dtype=np.float)
        x_new = np.array([0, 0.5], dtype=np.float)
        y_new = np.zeros(len(x_new), dtype=np.float)
        expected_y = np.array([0, 0], dtype=np.float)
        npt.assert_array_equal(stair_step(x, y, x_new, y_new), expected_y)

if __name__ == '__main__':
    unittest.main()
