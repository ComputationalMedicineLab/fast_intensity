# Copyright 2017 Thomas A. Lasko, Jacek Bajor

import datetime as dt
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from fast_intensity import FastRegression

class TestFastRegression(unittest.TestCase):
    def setUp(self):
        self.events = [100, 200, 300, 400, 500, 600]
        self.values = [10, 50, 100, 100, 10, 50]
        self.start = 100
        self.end = 600
        self.dates = [dt.date(2000, 1, 2), dt.date(2000, 1, 10),
                      dt.date(2000, 1, 15), dt.date(2000, 2, 1),
                      dt.date(2000, 2, 8), dt.date(2000, 2, 10)]
        self.date_times = [dt.datetime(2000, 1, 2), dt.datetime(2000, 1, 10),
                           dt.datetime(2000, 1, 15), dt.datetime(2000, 2, 1),
                           dt.datetime(2000, 2, 8), dt.datetime(2000, 2, 10)]
        self.date_strings = ['2000-01-02', '2000-01-10', '2000-01-15',
                             '2000-02-01', '2000-02-08', '2000-02-10']
        self.date_time_strings = ['2000-01-02 00:00:00', '2000-01-10 00:00:00',
                                  '2000-01-15 00:00:00', '2000-02-01 00:00:00',
                                  '2000-02-08 00:00:00', '2000-02-10 00:00:00']

    def test_accepts_list_as_input(self):
        fr = FastRegression(self.events, self.values, self.start, self.end)
        res = fr.run_inference()
        self.assertTrue(res.all())

    def test_accepts_array_as_input(self):
        fr = FastRegression(np.array(self.events), np.array(self.values),
                            self.start, self.end)
        res = fr.run_inference()
        self.assertTrue(res.all())

    def test_events_and_values_are_cut_if_out_of_bounds(self):
        events = [10, 50, 100, 150, 180, 200, 250, 320, 500]
        values = [100, 110, 120, 130, 140, 150, 160, 170, 180]
        start_event = 100
        end_event = 400
        expected_events = [100, 150, 180, 200, 250, 320]
        expected_values = [120, 130, 140, 150, 160, 170]

        fr = FastRegression(events, values, start_event, end_event)
        npt.assert_array_equal(fr.events, expected_events)
        npt.assert_array_equal(fr.values, expected_values)

    def test_accepts_date_as_input(self):
        start_date = dt.date(2000, 1, 1)
        end_date = dt.date(2000, 3, 1)
        fr = FastRegression.from_dates(self.dates, self.values, start_date,
                                       end_date)
        res = fr.run_inference()
        self.assertTrue(res.all())

    def test_accepts_datetime_as_input(self):
        start_date = dt.datetime(2000, 1, 1)
        end_date = dt.datetime(2000, 3, 1)
        fr = FastRegression.from_dates(self.date_times, self.values, start_date,
                                       end_date)
        res = fr.run_inference()
        self.assertTrue(res.all())

    def test_dates_are_cut_if_out_of_bounds(self):
        dates = [dt.date(2000, 1, 2), dt.date(2000, 1, 10),
                 dt.date(2000, 1, 15), dt.date(2000, 2, 1),
                 dt.date(2000, 2, 8), dt.date(2000, 2, 10)]

        start_date = dt.date(2000, 1, 5)
        end_date = dt.date(2000, 2, 5)
        expected_events = [5, 10, 27]

        fr = FastRegression.from_dates(self.dates, self.values, start_date,
                                       end_date)
        npt.assert_array_equal(fr.events, expected_events)

    def test_accepts_date_string_as_input(self):
        date_format = '%Y-%m-%d'
        start_date = '2000-01-01'
        end_date = '2000-03-01'
        fr = FastRegression.from_string_dates(self.date_strings, self.values,
                                              start_date, end_date,
                                              date_format=date_format)
        res = fr.run_inference()
        self.assertTrue((res).all())

    def test_accepts_date_time_string_as_input(self):
        date_format = '%Y-%m-%d %H:%M:%S'
        start_date = '2000-01-01 00:00:00'
        end_date = '2000-03-01 00:00:00'
        fr = FastRegression.from_string_dates(self.date_time_strings,
                                              self.values,start_date,
                                              end_date, date_format=date_format)
        res = fr.run_inference()
        self.assertTrue((res).all())

    def test_converts_dates_correctly(self):
        date_format = '%Y-%m-%d %H:%M:%S'
        start_date = '1999-04-01 00:00:00'
        end_date = '2010-11-08 00:00:00'
        date_time_strings = ['1999-04-01 00:00:00', '2003-08-18 00:00:00',
                             '2010-10-20 00:00:00', '2010-11-08 00:00:00']
        values = [100, 200, 150, 300]
        expected_events = [0, 1600, 4220, 4239]

        fr = FastRegression.from_string_dates(date_time_strings, values,
                                              start_date, end_date,
                                              date_format=date_format)

        npt.assert_array_equal(fr.events, expected_events)

    def test_converts_dates_and_adds_endoints_correctly(self):
        date_format = '%Y-%m-%d'
        start_date = '1999-03-31'
        end_date = '2010-11-10'
        date_time_strings = ['1999-04-01', '2003-08-18', '2010-10-20',
                             '2010-11-08']
        values = [100, 200, 150, 300]
        expected_events = [1, 1601, 4221, 4240]

        fr = FastRegression.from_string_dates(date_time_strings, values,
                                              start_date, end_date,
                                              date_format=date_format)

        npt.assert_array_equal(fr.events, expected_events)

    def test_raises_exception_on_events_values_diff_len(self):
        with self.assertRaises(ValueError):
            fr = FastRegression(self.events, [1,2], self.start, self.end)

    def test_raises_exception_on_empty_events_values(self):
        with self.assertRaises(ValueError):
            fr = FastRegression([], [], self.start, self.end)

    def test_constant_for_one_event_value(self):
        fr = FastRegression([200], [100], self.start, self.end)
        result = fr.run_inference()
        self.assertTrue((result == 100).all())

    def test_has_one_grid_point_for_equal_start_end_date(self):
        fr = FastRegression([200], [100], 200, 200)
        result = fr.run_inference()
        npt.assert_array_equal(len(fr.grid), 1)

if __name__ == '__main__':
    unittest.main()
