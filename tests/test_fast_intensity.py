# Copyright 2017 Thomas A. Lasko, Jacek Bajor
import unittest

import numpy as np
import numpy.testing as npt

from fast_intensity import FastIntensity


class TestFastIntensity(unittest.TestCase):

    def setUp(self):
        self.events = [1937, 2279, 2364, 3876, 4011]
        self.grid = np.linspace(-1, 4240, num=5)
        self.start = -1
        self.end = 4240

    def test_accepts_list_as_input(self):
        fi = FastIntensity(self.events, self.grid)
        res = fi.run_inference()
        self.assertTrue((res >= 0).all())

    def test_accepts_array_as_input(self):
        fi = FastIntensity(np.array(self.events), self.grid)
        res = fi.run_inference()
        self.assertTrue((res >= 0).all())

    def test_events_are_cut_if_out_of_bounds(self):
        events = [1937,1939,1945,1979,1986,2200,2026,2029,2189,2211,2213,2214]
        grid = np.linspace(2000, 2200, num=4)
        expected_events = [2200,2026,2029,2189]

        fi = FastIntensity(events, grid)
        npt.assert_array_equal(fi.events, expected_events)
