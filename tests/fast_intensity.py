# Copyright 2017 Thomas A. Lasko, Jacek Bajor

import datetime as dt
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from fast_intensity import FastIntensity

@unittest.skip('API in flux')
class TestFastIntensity(unittest.TestCase):
    def setUp(self):
        self.events = [1937, 2279, 2364, 3876, 4011]
        self.start = -1
        self.end = 4240
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
        fi = FastIntensity(self.events, self.start, self.end)
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_accepts_array_as_input(self):
        fi = FastIntensity(np.array(self.events), self.start, self.end)
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_events_are_cut_if_out_of_bounds(self):
        events = [1937,1939,1945,1979,1986,2200,2026,2029,2189,2211,2213,2214]
        start_event = 2000
        end_event = 2200
        expected_events = [2200,2026,2029,2189]

        fi = FastIntensity(events, start_event, end_event)
        npt.assert_array_equal(fi.events, expected_events)

    def test_accepts_date_as_input(self):
        start_date = dt.date(2000, 1, 1)
        end_date = dt.date(2000, 3, 1)
        fi = FastIntensity.from_dates(self.dates, start_date, end_date)
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_accepts_datetime_as_input(self):
        start_date = dt.datetime(2000, 1, 1)
        end_date = dt.datetime(2000, 3, 1)
        fi = FastIntensity.from_dates(self.date_times, start_date, end_date)
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_dates_are_cut_if_out_of_bounds(self):
        self.dates = [dt.date(2000, 1, 2), dt.date(2000, 1, 10),
                      dt.date(2000, 1, 15), dt.date(2000, 2, 1),
                      dt.date(2000, 2, 8), dt.date(2000, 2, 10)]

        start_date = dt.date(2000, 1, 5)
        end_date = dt.date(2000, 2, 5)
        expected_events = [5, 10, 27]

        fi = FastIntensity.from_dates(self.dates, start_date, end_date)
        npt.assert_array_equal(fi.events, expected_events)

    def test_accepts_date_string_as_input(self):
        date_format = '%Y-%m-%d'
        start_date = '2000-01-01'
        end_date = '2000-03-01'
        fi = FastIntensity.from_string_dates(self.date_strings, start_date,
                                             end_date, date_format=date_format)
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_accepts_date_time_string_as_input(self):
        date_format = '%Y-%m-%d %H:%M:%S'
        start_date = '2000-01-01 00:00:00'
        end_date = '2000-03-01 00:00:00'
        fi = FastIntensity.from_string_dates(self.date_time_strings, start_date,
                                             end_date, date_format=date_format)
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_converts_dates_correctly(self):
        date_format = '%Y-%m-%d %H:%M:%S'
        start_date = '1999-04-01 00:00:00'
        end_date = '2010-11-08 00:00:00'
        date_time_strings = ['1999-04-01 00:00:00', '2003-08-18 00:00:00',
                             '2010-10-20 00:00:00', '2010-11-08 00:00:00']
        expected_events = [0, 1600, 4220, 4239]

        fi = FastIntensity.from_string_dates(date_time_strings, start_date,
                                             end_date, date_format=date_format)

        npt.assert_array_equal(fi.events, expected_events)

    def test_accepts_npdatetime_as_input_and_converts_correctly(self):
        dates = np.array([np.datetime64('1999-04-01'),
                          np.datetime64('2003-08-18T00:00:00.000000000'),
                          np.datetime64('2010-10-20T00:00:00'),
                          np.datetime64('2010-11-08')])
        start_date = np.datetime64('1999-04-01T00:00:00')
        end_date = np.datetime64('2010-11-08T00:00:00')
        expected_events = [0, 1600, 4220, 4239]

        fi = FastIntensity.from_dates(dates, start_date, end_date)
        npt.assert_array_equal(fi.events, expected_events)

if __name__ == '__main__':
    unittest.main()
