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
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        x_new = [2.5, 4.5]
        expected_y = [2, 4]
        npt.assert_array_equal(stair_step(x, y, x_new, [None, None]), expected_y)

    def test_returns_correct_interpolation_for_decreasing_vals(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        x_new = [2.5, 4.5]
        expected_y = [4, 2]
        npt.assert_array_equal(stair_step(x, y, x_new, [None, None]), expected_y)

    def test_raises_exception_when_x_unsorted(self):
        x = [5, 1, 6, 2, 7]
        y = [1, 2, 3, 4, 5]
        x_new = [2.5, 4.5]
        self.assertRaises(Exception, stair_step, (x, y, x_new, [None, None]))

    def test_raises_exception_when_x_and_y_different_length(self):
        x = [1, 2, 3, 4]
        y = [1, 2, 3, 4, 5, 6]
        x_new = [2.5, 4.5]
        self.assertRaises(Exception, stair_step, (x, y, x_new, [None, None]))

    def test_returns_correct_interpolation_for_right_out_of_bounds(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        x_new = [5, 6]
        expected_y = [1, 1]
        npt.assert_array_equal(stair_step(x, y, x_new, [None, None]), expected_y)

    def test_returns_nan_for_left_out_of_bounds(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        x_new = [0, 0.5]
        expected_y = [np.nan, np.nan]
        npt.assert_array_equal(stair_step(x, y, x_new, [None, None]), expected_y)

if __name__ == '__main__':
    unittest.main()
