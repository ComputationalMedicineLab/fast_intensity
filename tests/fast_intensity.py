import datetime as dt
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from fast_intensity import FastIntensity

class TestFastIntensity(unittest.TestCase):
    def setUp(self):
        self.events_with_endpoints = [-1,1937,1939,1945,1979,1986,2026,2029,
            2189,2211,2212,2213,2214,2216,2226,2238,2240,2243,2247,2279,2364,
            2369,2380,2408,2412,2413,2420,2450,2457,2541,2590,2642,2672,2701,
            2803,2971,3010,3153,3295,3336,3395,3625,3659,3723,3766,3876,4011,
            4240]
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
        fi = FastIntensity(self.events_with_endpoints)
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_accepts_array_as_input(self):
        fi = FastIntensity(np.array(self.events_with_endpoints))
        res = fi.run_inference(5)
        self.assertTrue((res >= 0).all())

    def test_base_and_event_sources_have_the_same_state(self):
        events = self.events_with_endpoints[1:-1]
        start = self.events_with_endpoints[0]
        end = self.events_with_endpoints[-1]

        fi_1 = FastIntensity(self.events_with_endpoints)
        fi_2 = FastIntensity.from_events(events, start, end)

        npt.assert_array_equal(fi_1.events_with_endpoints,
                               fi_2.events_with_endpoints)

    def test_events_are_cut_if_out_of_bounds(self):
        events = [1937,1939,1945,1979,1986,2026,2029,2189,2211,2212,2213,2214]
        start_event = 2000
        end_event = 2200
        expected_events_wt = [2000,2026,2029,2189,2200]

        fi = FastIntensity.from_events(events, start_event, end_event)
        npt.assert_array_equal(fi.events_with_endpoints, expected_events_wt)

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
        expected_events_wt = [0, 5, 10, 27, 31]

        fi = FastIntensity.from_dates(self.dates, start_date, end_date)
        npt.assert_array_equal(fi.events_with_endpoints, expected_events_wt)

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
        expected_events_wt = [0, 1600, 4220, 4239]

        fi = FastIntensity.from_string_dates(date_time_strings, start_date,
                                             end_date, date_format=date_format)

        npt.assert_array_equal(fi.events_with_endpoints, expected_events_wt)

    def test_converts_dates_and_adds_endoints_correctly(self):
        date_format = '%Y-%m-%d'
        start_date = '1999-03-31'
        end_date = '2010-11-10'
        date_time_strings = ['1999-04-01', '2003-08-18', '2010-10-20', '2010-11-08']
        expected_events_wt = [0, 1, 1601, 4221, 4240, 4242]

        fi = FastIntensity.from_string_dates(date_time_strings, start_date,
                                             end_date, date_format=date_format)

        npt.assert_array_equal(fi.events_with_endpoints, expected_events_wt)

if __name__ == '__main__':
    unittest.main()
