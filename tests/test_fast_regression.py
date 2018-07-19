# Copyright 2017 Thomas A. Lasko, Jacek Bajor
import unittest

import numpy as np
import numpy.testing as npt

from fast_intensity import FastRegression


class TestFastRegression(unittest.TestCase):

    def setUp(self):
        self.events = [100, 200, 300, 400, 500, 600]
        self.values = [10, 50, 100, 100, 10, 50]
        self.grid = np.linspace(100, 600, num=6)
        self.start = 100
        self.end = 600

    def test_accepts_list_as_input(self):
        fr = FastRegression(self.events, self.values, self.grid)
        res = fr.run_inference()
        self.assertTrue(res.all())

    def test_accepts_array_as_input(self):
        fr = FastRegression(np.array(self.events),
                            np.array(self.values),
                            self.grid)
        res = fr.run_inference()
        self.assertTrue(res.all())

    def test_events_and_values_are_cut_if_out_of_bounds(self):
        events = [10, 50, 100, 150, 180, 200, 250, 320, 500]
        values = [100, 110, 120, 130, 140, 150, 160, 170, 180]
        expected_events = [100, 150, 180, 200, 250, 320]
        expected_values = [120, 130, 140, 150, 160, 170]
        grid = np.linspace(100, 400, num=6)

        fr = FastRegression(events, values, grid)
        npt.assert_array_equal(fr.events, expected_events)
        npt.assert_array_equal(fr.values, expected_values)

    def test_raises_exception_on_events_values_diff_len(self):
        with self.assertRaises(ValueError):
            fr = FastRegression(self.events, [1,2], self.grid)

    def test_raises_exception_on_empty_events_values(self):
        with self.assertRaises(ValueError):
            fr = FastRegression([], [], self.grid)

    def test_constant_for_one_event_value(self):
        fr = FastRegression([200], [100], self.grid)
        result = fr.run_inference()
        self.assertTrue((result == 100).all())

    def test_has_one_grid_point_for_single_value_grid(self):
        fr = FastRegression([200], [100], [200])
        result = fr.run_inference()
        npt.assert_array_equal(len(fr.grid), 1)
