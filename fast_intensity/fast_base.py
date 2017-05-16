# Copyright 2017 Thomas A. Lasko, Jacek Bajor

from datetime import datetime

import numpy as np

class FastBase(object):

    @staticmethod
    def time_delta_in_days(a, b):
        """
        Return time difference in days.

        Args:
            a, b (date or datetime)

        Returns:
            float: time difference in days (exact, not rounded)
        """
        return (a-b).total_seconds()/(24*60*60)

    @staticmethod
    def convert_string_dates_to_events(string_dates, start_date, end_date,
                                       date_format='%Y-%m-%d %H:%M:%S'):
        start_date = datetime.strptime(start_date, date_format)
        end_date = datetime.strptime(end_date, date_format)
        dates = [ datetime.strptime(d, date_format) for d in string_dates]
        return FastBase.convert_dates_to_events(dates, start_date, end_date)

    @staticmethod
    def convert_dates_to_events(dates, start_date, end_date):
        """
        Convert dates to events.

        Args:
            dates (array-like of date/datetime)
            start_event (date/datetime)

        Returns:
            list of numbers representing events
        """
        events = np.zeros(len(dates))

        if isinstance(dates, np.ndarray) and np.issubdtype(dates.dtype, np.datetime64):
            dates = dates.astype('M8[s]').astype(datetime)

        if np.issubdtype(type(start_date), np.datetime64):
            start_date = start_date.astype('M8[s]').astype(datetime)

        if np.issubdtype(type(end_date), np.datetime64):
            end_date = end_date.astype('M8[s]').astype(datetime)

        for i, d in enumerate(dates):
            events[i] = FastBase.time_delta_in_days(d, start_date)
        return events, 0, FastBase.time_delta_in_days(end_date, start_date)

    def _generate_grid(self, resolution, density):
        """
        Generate grid (x-axis).

        Args:
            density (number): average number of bin edges between neighboring
                points (default 1/365).
            resolution (number): resolution for bin edges in units of days
                (default 1).

        Returns:
            np.array of evenly spaced numerical values
        """
        grid_len = np.max([int(np.round((self.end - self.start) / resolution)), 1])

        return np.linspace(self.start, self.end, grid_len)
