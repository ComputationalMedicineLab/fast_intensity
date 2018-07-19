# Copyright 2017 Thomas A. Lasko, Jacek Bajor
import numpy as np
from scipy.interpolate import pchip_interpolate

from .fast_base import FastBase


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

    if len(grid) > 1:
        f_event, f_value = ([grid[0]], [values[0]]
                            ) if grid[0] != events[0] else ([], [])
        l_event, l_value = ([grid[-1]], [values[-1]]
                            ) if grid[-1] != events[-1] else ([], [])
        events = np.concatenate((f_event, events, l_event))
        values = np.concatenate((f_value, values, l_value))

    return pchip_interpolate(events, values, grid)
