# Copyright 2017 Thomas A. Lasko, Jacek Bajor

from .fast_base import FastBase

from datetime import datetime

import numpy as np
import numpy.random as npr
import warnings

# Import and compile Cython files
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from .stair_step import stair_step
from .fast_hist import fast_hist


class FastIntensity(FastBase):
    """Estimates (potentially nonstationary) event intensity vs. time.

    This class uses Completely Random Average Shifted Histograms (CRASH) to
    compute a continuous curve of event  intensity vs. time, as described in **
    Citation TBD **.

    Each histogram is defined by a random number of bin edges, with the
    location of each bin edge sampled uniformly at random between event
    *indices* (not their locations). For example, with the sequence of events
    [1, 2, 3, 100], there is the same probability that an edge will appear between
    3 and 3 as between 3 and 100.  This allows for the final density estimation
    to adapt its bandwidth to the nonstationarity of event locations. A
    constraint on the minimum number of events per bin keeps density peaks from
    forming pathologically around each event and at endpoints.

    Attributes:
        events (array-like of sorted real numbers): event times
        grid (np.array of sorted real numbers): timepoints at which the
            intensity curve is computed
        min_count: The minimum number of points per bin.

    Usage:
        events = [10, 15, 16, 17, 28]
        grid = np.arange(0, 50, 1)

        fi = FastIntensity(events, grid=grid)
        intensity = fi.run_inference()
    """

    def __init__(self, events, grid, iterations=100, min_count=3):
        """
        Initialize with events and inference parameters.

        Args:
            events (array-like of sorted reals): event times
            grid (array-like of sorted reals): time points at which to
                compute the intensity curve.
            iterations: number of inference iterations (default 100).
            min_count: The minimum number of events per bin (default 3).
        """
        # Cut out of bounds values
        before_start = np.where(events < grid[0])
        events = np.delete(events, before_start)

        after_end = np.where(events > grid[-1])
        events = np.delete(events, after_end)

        self.events = events
        self.grid = grid
        self.iterations = iterations

    def run_inference(self):
        """Run event intensity inference.

        Returns:
            np.array of event intensity, calculated at times defined by
              self.grid, with units of events per time.
        """
        meanvals = np.zeros(len(self.grid))
        vals = np.zeros(len(self.grid), dtype=np.float)
        n = len(self.events) + 1
        min_count = 3

        # Compute event_indices once for all iterations of _get_boundaries, for
        # efficiency. (This has a measurable effect on run time.)
        self.event_indices = np.linspace(0, n, n + 1)

        self._events_w_endpoints = np.concatenate(([self.start], self.events,
                                                   [self.end]))
        max_bins = int(self.event_indices[-1] // min_count)

        for i in range(self.iterations):
            if max_bins < 2:
                num_bins = 1
            else:
                # randint high value is exclusive
                num_bins = npr.randint(1, max_bins + 1)

            boundaries = self._get_boundaries(num_bins, min_count)
            h = fast_hist(self.events, boundaries)
            vals = stair_step(boundaries, h, self.grid, vals)
            meanvals = meanvals + (vals - meanvals) / (i + 1)

        return meanvals

    def _get_boundaries(self, num_bins, min_count):
        """Compute random bin boundaries for histogram, respecting min_count.

        Boundaries are sampled uniformly at random in sequence space, with the
        constraint that all bins have at least min_count events in them
        (with endpoints considered events). This means, that a boundary is
        equally likely to occur between any two events, regardless of the
        spacing of those events, so long as min_count is respected. This
        tends to give a smoothness to the final density estimation that varies
        appropriately with the density of events.

        Args:
            num_bins: The number of bins to be defined by the boundaries.
            min_count: The minimum number of events (including
              endpoints) to be present in any bin.

        Returns:
            np.array of new bin boundaries
        """
        sequence_boundaries = _get_sequence_boundaries(
            num_bins, num_events=len(self.events), min_count=min_count)

        data_boundaries = np.interp(
            sequence_boundaries, self.event_indices, self._events_w_endpoints)

        return data_boundaries


def _get_sequence_boundaries(num_bins, num_events, min_count=3):
    """Compute the bin boundaries in (0-based) sequence index space.

    For example, a boundary at 0.35 means that the boundary is 35% of the way
    between the beginning boundary (before any events) and the first
    event. (Although that particular boundary cannot exist unless min_count=1.)
    Boundaries are sampled uniformly at random in sequence space, subject to
    the constraint that all boundaries are separated by at least min_count.

    For efficiency, no checking is done to ensure that the arguments are
    consistent. Setting num_bins=1 will always return valid boundaries for
    num_events > 0. Inconsistent combinations as defined below result in
    undefined behavior.

    Args:
        num_bins: The number of bins to be defined by the boundaries. Must
          satisfy num_bins <= ((num_events + 2) / min_count) or undefined
          behavior results.
        num_events: The positive number of events to be binned (not counting
          the overall start and end boundaries as events). Must be positive
          (and nonzero) or underfine behavior results.
        min_count: The minimum number of indices between boundaries.

    Returns:
       np.array of bin boundaries.
    """

    # The bin at each end must contain at least pad = min_count - 1 actual
    # events, because the endpoints count as an included event, even though the
    # intervals stop exactly at that event.
    start = 0
    end = num_events + 1
    pad = min_count - 1

    boundaries = np.empty(num_bins + 1, dtype='float')
    boundaries[0] = start
    boundaries[-1] = end
    if num_bins == 1:
        return boundaries

    boundaries[1:-1] = np.arange(start=pad, stop=min_count * (num_bins - 1),
                                 step=min_count, dtype='float')
    slop = npr.uniform(low=0, high=end - pad -
                       boundaries[-2], size=num_bins - 1)
    slop.sort()
    np.add(boundaries[1:-1], slop, out=boundaries[1:-1])
    return boundaries
