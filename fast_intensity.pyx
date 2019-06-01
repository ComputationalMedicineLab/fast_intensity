"""Fast intensity inference"""
import numpy as np
cimport numpy as np
import numpy.random as npr
from scipy.interpolate import pchip_interpolate

ctypedef np.float_t DTYPE_t

__version__ = '0.4.dev0'
__all__ = ['fast_hist', 'stair_step', 'FastIntensity', 'FastRegression']


class FastBase(object):
    def __init__(self, grid):
        """Initialize with parameters.

        Args:
            grid (array-like of reals): Evenly spaced time points at which to
                compute the curve.
        """
        self.grid = grid

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


def fast_hist(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] edges):
    """
    Return density histogram.

    Calculates density of elements x in bins defined by edges. Assumes values
    and edges are sorted, and edges[0] < x < edges[-1]. Behavior for unsorted
    values is undefined.

    Args:
        x (np.array of np.float numbers): values
        edges (np.array of np.float numbers): bin edges (2 or more values)

    Returns:
        np.array of density histogram (float)
    """
    cdef np.ndarray density = np.zeros(len(edges) - 1, dtype=np.float)
    cdef int n = len(x)
    cdef int i = 0
    cdef int j = 1
    cdef int start = i

    while i < n:
        start = i

        while x[i] > edges[j]:
            j = j + 1

        while i < n and x[i] <= edges[j]:
            i = i + 1

        edges_distance = (edges[j] - edges[j - 1])

        if edges_distance != 0:
            density[j-1] = (i - start) / edges_distance

    return density


def stair_step(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y,
               np.ndarray[DTYPE_t, ndim=1] xp, np.ndarray[DTYPE_t, ndim=1] yp):
    """
    Previous neighbor interpolation. Behavoir undefined for unsorted points.

    Args:
        x (np.array of np.float numbers): sample points, sorted.
        y (np.array of np.float numbers): sample values (same size as x)
        xp (np.array of np.float numbers): query points
        yp (np.array of np.float numbers): preallocated list or np.array for
            query values (same size as xp)

    Returns:
        np.array of interpolated values (float)
    """
    cdef int n = xp.shape[0]
    cdef int m = y.shape[0]
    cdef int j = 0
    cdef int i = 0

    while j < n and xp[j] < x[i]:
        yp[j] = 0
        j += 1

    while j < n and i < m:
        while i < m-1 and xp[j] >= x[i+1]:
            i += 1
        yp[j] = y[i]
        j += 1

    return yp


class FastIntensity(FastBase):
    """Estimates (potentially nonstationary) event intensity vs. time.

    This class uses Completely Random Average Shifted Histograms (CRASH) to
    compute a continuous curve of event intensity vs. time, as described in **
    Citation TBD **.

    Each histogram is defined by a random number of bin edges, with the
    location of each bin edge sampled uniformly at random between event
    *indices* (not their locations). For example, with the sequence of events
    [1, 2, 3, 100], there is the same probability that an edge will appear
    between 3 and 3 as between 3 and 100.  This allows for the final density
    estimation to adapt its bandwidth to the nonstationarity of event
    locations. A constraint on the minimum number of events per bin keeps
    density peaks from forming pathologically around each event and at
    endpoints.

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
            # fast_hist expects arrays of floats; in some scenarios events may
            # be an array of longs.  We explicitly cast here to avoid a buffer
            # data type mismatch
            h = fast_hist(self.events.astype(np.float),
                          boundaries.astype(np.float))
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

        events = np.array(events, dtype=np.float)
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

        return _pchip_with_const_extrapolation(self.events,
                                               self.values,
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
