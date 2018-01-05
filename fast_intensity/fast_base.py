# Copyright 2017 Thomas A. Lasko, Jacek Bajor

from datetime import datetime
from scipy.interpolate import pchip_interpolate

import numpy as np
import pdb


class FastBase(object):
    def __init__(self, events, start_time, end_time, resolution):
        """
        Initialize with parameters.

        Args:
            events (array-like of real numbers): event times in units of days
                since an arbitrary reference point
            start_time (number): beginning of the computed inference time range
            end_time (number): end of the computed inference time range
            resolution (number): grid inference resolution in units of days.
        """
        events = np.array(events, dtype=np.float)
        # Cut out of bounds events
        events = np.delete(events, np.where(events < start_time))
        events = np.delete(events, np.where(events > end_time))

        self.events = events
        self.start = start_time
        self.end = end_time
        self.resolution = resolution
        self.grid = _generate_grid(self.start, self.end, self.resolution)

    def _pchip_with_const_extrapolation(self, events, values, grid):
        """
        Interpolates between readings, extrapolates based on boundry values.
        """

        if len(grid) > 1:
            f_event, f_value = ([grid[0]], [values[0]]
                                ) if grid[0] != events[0] else ([], [])
            l_event, l_value = ([grid[-1]], [values[-1]]
                                ) if grid[-1] != events[-1] else ([], [])
            events = np.concatenate((f_event, events, l_event))
            values = np.concatenate((f_value, values, l_value))

        return pchip_interpolate(events, values, grid)

    def run_inference(self):
        raise NotImplementedError

    @staticmethod
    def time_delta_in_days(a, b):
        """
        Return time difference in days.

        Args:
            a, b (date or datetime)

        Returns:
            float: time difference in days (exact, not rounded)
        """
        return (a - b).total_seconds() / (24 * 60 * 60)

    @staticmethod
    def convert_string_dates_to_events(string_dates, start_date, end_date,
                                       date_format='%Y-%m-%d %H:%M:%S'):
        start_date = datetime.strptime(start_date, date_format)
        end_date = datetime.strptime(end_date, date_format)
        dates = [datetime.strptime(d, date_format) for d in string_dates]
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


def _generate_grid(start, end, resolution=1):
    """Generate evenly spaced grid with approximate resolution.

    Args:
        start (number): starting value
        end (number): ending value
        resolution (number): desired spacing for grid points in units of
            days (default 1).

    Returns: np.array of values, evenly spaced approximately by 'resolution'.

    """
    n = np.max([int(np.round((end - start) / resolution)), 1])
    return np.linspace(start, end, n)
