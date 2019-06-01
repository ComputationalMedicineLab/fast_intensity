# cython: language_level=3
import unittest

import numpy as np
import numpy.testing as npt
from fast_intensity import *
from fast_intensity cimport density_hist, stair_step


class TestFastHistFunction(unittest.TestCase):
    def test_returns_correct_density_values_for_uniform_data(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)
        edges = np.array([0, 5.5, 11], dtype=np.float)
        result = density_hist(x, edges)
        npt.assert_array_equal(result[0], result[1])

    def test_returns_correct_density_values_for_one_val(self):
        x = np.array([5], dtype=np.float)
        edges = np.array([4, 6], dtype=np.float)
        self.assertEqual(density_hist(x, edges), np.array([0.5]))


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


class TestIntensity(unittest.TestCase):
    def setUp(self):
        self.events = [1937, 2279, 2364, 3876, 4011]
        self.grid = np.linspace(-1, 4240, num=5)
        self.start = -1
        self.end = 4240

    def test_accepts_list_as_input(self):
        res = infer_intensity(self.events, self.grid)
        self.assertTrue((res >= 0).all())

    def test_accepts_array_as_input(self):
        res = infer_intensity(np.array(self.events), self.grid)
        self.assertTrue((res >= 0).all())

    # def test_events_are_cut_if_out_of_bounds(self):
    #     events = [1937,1939,1945,1979,1986,2200,2026,2029,2189,2211,2213,2214]
    #     grid = np.linspace(2000, 2200, num=4)
    #     expected_events = [2200,2026,2029,2189]

    #     fi = FastIntensity(events, grid)
    #     npt.assert_array_equal(fi.events, expected_events)


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.events = [100, 200, 300, 400, 500, 600]
        self.values = [10, 50, 100, 100, 10, 50]
        self.grid = np.linspace(100, 600, num=6)
        self.start = 100
        self.end = 600

    def test_accepts_list_as_input(self):
        res = regression(self.events, self.values, self.grid)
        self.assertTrue(res.all())

    def test_accepts_array_as_input(self):
        res = regression(np.array(self.events),
                         np.array(self.values),
                         self.grid)
        self.assertTrue(res.all())

    # def test_events_and_values_are_cut_if_out_of_bounds(self):
    #     events = [10, 50, 100, 150, 180, 200, 250, 320, 500]
    #     values = [100, 110, 120, 130, 140, 150, 160, 170, 180]
    #     expected_events = [100, 150, 180, 200, 250, 320]
    #     expected_values = [120, 130, 140, 150, 160, 170]
    #     grid = np.linspace(100, 400, num=6)

    #     fr = FastRegression(events, values, grid)
    #     npt.assert_array_equal(fr.events, expected_events)
    #     npt.assert_array_equal(fr.values, expected_values)

    def test_raises_exception_on_events_values_diff_len(self):
        with self.assertRaises(ValueError):
            regression(self.events, [1,2], self.grid)

    def test_raises_exception_on_empty_events_values(self):
        with self.assertRaises(ValueError):
            regression([], [], self.grid)

    def test_constant_for_one_event_value(self):
        res = regression([200], [100], self.grid)
        self.assertTrue(np.all(res == 100))

    # def test_has_one_grid_point_for_single_value_grid(self):
    #     fr = FastRegression([200], [100], [200])
    #     result = fr.run_inference()
    #     npt.assert_array_equal(len(fr.grid), 1)
