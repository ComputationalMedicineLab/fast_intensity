import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from fast_intensity.fast_hist import fast_hist

class TestFastHistFunction(unittest.TestCase):
    def setUp(self):
        pass

    def test_returns_correct_density_values(self):
        pass

    def test_raises_exception_when_edges_unsorted(self):
        pass

    def test_raises_exception_when_values_unsorted(self):
        pass

if __name__ == '__main__':
    unittest.main()
