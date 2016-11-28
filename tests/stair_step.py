import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from fast_intensity.stair_step import stair_step

class TestStairStepFunction(unittest.TestCase):
    def setUp(self):
        pass

    def test_returns_correct_interpolation_for_increasing_vals(self):
        pass

    def test_returns_correct_interpolation_for_decreasing_vals(self):
        pass

    def test_raises_exception_when_values_unsorted(self):
        pass

    def test_raises_warning_when_values_out_of_bounds(self):
        pass

if __name__ == '__main__':
    unittest.main()
