# Copyright 2017 Thomas A. Lasko, Jacek Bajor

from datetime import datetime

import numpy as np
import pdb


class FastBase(object):
    def __init__(self, events, grid):
        """Initialize with parameters.

        Args:
            events (array-like of reals): event times in units of days
                since an arbitrary reference point
            grid (array-like of reals): Evenly spaced time points at which to
                compute the curve.
        """
        self.grid = grid

        events = np.array(events, dtype=np.float)
        # Cut out of bounds events
        events = np.delete(events, np.where(events < grid[0]))
        events = np.delete(events, np.where(events > grid[-1]))

        self.events = events

    @property
    def start(self):
        return self.grid[0]

    @property
    def end(self):
        return self.grid[-1]

    @property
    def resolution(self):
        return self.grid[1] - self.grid[0]

    def run_inference(self):
        raise NotImplementedError

#     @staticmethod
#     def time_delta_in_days(a, b):
#         """
#         Return time difference in days.

#         Args:
#             a, b (date or datetime)

#         Returns:
#             float: time difference in days (exact, not rounded)
#         """
#         return (a - b).total_seconds() / (24 * 60 * 60)

#     @staticmethod
#     def convert_string_dates_to_events(string_dates, start_date, end_date,
#                                        date_format='%Y-%m-%d %H:%M:%S'):
#         start_date = datetime.strptime(start_date, date_format)
#         end_date = datetime.strptime(end_date, date_format)
#         dates = [datetime.strptime(d, date_format) for d in string_dates]
#         return FastBase.convert_dates_to_events(dates, start_date, end_date)

#     @staticmethod
#     def convert_dates_to_events(dates, start_date, end_date):
#         """
#         Convert dates to events.

#         Args:
#             dates (array-like of date/datetime)
#             start_event (date/datetime)

#         Returns:
#             list of numbers representing events
#         """
#         events = np.zeros(len(dates))

#         if isinstance(dates, np.ndarray) and np.issubdtype(dates.dtype, np.datetime64):
#             dates = dates.astype('M8[s]').astype(datetime)

#         if np.issubdtype(type(start_date), np.datetime64):
#             start_date = start_date.astype('M8[s]').astype(datetime)

#         if np.issubdtype(type(end_date), np.datetime64):
#             end_date = end_date.astype('M8[s]').astype(datetime)

#         for i, d in enumerate(dates):
#             events[i] = FastBase.time_delta_in_days(d, start_date)
#         return events, 0, FastBase.time_delta_in_days(end_date, start_date)
