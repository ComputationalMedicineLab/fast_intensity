# Copyright 2017 Thomas A. Lasko, Jacek Bajor
import pdb
from .fast_base import FastBase

from datetime import datetime
from scipy.interpolate import pchip_interpolate

import numpy as np


class FastRegression(FastBase):
    """Estimates values over time.

    Attributes:
        events (array-like of real numbers): event times in units of days
            since an arbitrary reference point
        values (array-like of real numbers): values for each event time
        start (number): beginning of the computed inference time range
        end (number): end of the computed inference time range
        grid (np.array of real numbers): evenly spaced timepoints at which the
            curve is computed

    Usage:
        events = [10, 15, 16, 17, 28]
        values = [120, 128, 119, 148, 144]
        grid = np.arange(0, 50, 1)

        fr = FastRegression(events, values, grid)
        regression = fr.run_inference()

        # dates = [dt.datetime(2000, 1, 2), dt.datetime(2000, 1, 10),
        #          dt.datetime(2000, 1, 15), dt.datetime(2000, 2, 1)]

        # fr = FastRegression.from_dates(dates, values,
        #                                start_date=dt.datetime(2000, 1, 1),
        #                                end_date=dt.datetime(2000, 3, 1))
        # regression = fr.run_inference()

        # date_strs = ['2000-01-02', '2000-01-10', '2000-01-15', '2000-02-01']

        # fr = FastRegression.from_string_dates(date_strs, values,
        #                                       start_date='2000-01-01',
        #                                       end_date='2000-03-01',
        #                                       date_format='%Y-%m-%d')
        # regression = fr.run_inference()
    """

    def __init__(self, events, values, grid):
        """
        Initialize with events and corresponding values.

        Args:
            events (array-like of real numbers): event times in units of days
                since an arbitrary reference point
            values (array-like of real numbers): values for corresponding event
                times, must be same the length as events
            grid (array-like of reals): Evenly spaced time points at which to
                compute the curve.
        """
        if len(events) != len(values):
            raise ValueError("Events and values are different lengths.")

        if len(events) == 0:
            raise ValueError("Events and values are empty.")

        values = np.array(values, dtype=np.float)

        # Cut out of bounds values
        before_start = np.where(events < grid[0])
        values = np.delete(values, before_start)
        events = np.delete(events, before_start)

        after_end = np.where(events > grid[-1])
        values = np.delete(values, after_end)
        events = np.delete(events, after_end)

        self.events = events
        self.values = values
        self.grid = grid

    # @classmethod
    # def from_dates(cls, dates, values, start_date, end_date, resolution=1):
    #     """
    #     Convert dates/datetimes to events and initialize an instance.

    #     Args:
    #         dates (array-like of date/datetime)
    #         values (array-like of real numbers)
    #         start_event (date/datetime)
    #         end_event (date/datetime)
    #     """
    #     events, start_e, end_e = FastBase.convert_dates_to_events(
    #         dates, start_date, end_date)

    #     return cls(events, values, start_e, end_e, resolution)

    # @classmethod
    # def from_string_dates(cls, dates, values, start_date, end_date,
    #                       date_format='%Y-%m-%d %H:%M:%S', resolution=1):
    #     """
    #     Convert date strings to events and initialize an instance.

    #     Args:
    #         dates (array-like of strings): dates represented by correctly
    #             formatted strings
    #         values (array-like of real numbers): values for corresponding dates,
    #             must be same the length as events
    #         start_event (string): date represented by a correctly formatted
    #             string
    #         end_event (string): date represented by a correctly formatted
    #             string
    #         date_format (string): format of dates in the input (same as used
    #             in datetime.datetime.strptime() function)
    #     """
    #     events, start_e, end_e = FastBase.convert_string_dates_to_events(
    #         dates, start_date, end_date, date_format)

    #     return cls(events, values, start_e, end_e, resolution)

    def run_inference(self):
        """
        Run regression inference.

        Returns
            np.array
        """
        if len(self.events) == 1:
            return np.ones(len(self.grid)) * self.values[0]

        return _pchip_with_const_extrapolation(self.events, self.values,
                                               self.grid)


def _pchip_with_const_extrapolation(events, values, grid):
    """
    Interpolates between readings, extrapolates based on boundry values.
    """
    # TODO: Change to use mean outside of the boundaries, or maybe to approach
    # the mean at the endpoints of grid.
    if len(grid) > 1:
        f_event, f_value = ([grid[0]], [values[0]]
                            ) if grid[0] != events[0] else ([], [])
        l_event, l_value = ([grid[-1]], [values[-1]]
                            ) if grid[-1] != events[-1] else ([], [])
        events = np.concatenate((f_event, events, l_event))
        values = np.concatenate((f_value, values, l_value))

    return pchip_interpolate(events, values, grid)
