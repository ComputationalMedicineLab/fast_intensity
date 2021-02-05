# Compiler directives have to come before code, but may come after comments
# cython: language_level=3
"""Fast intensity inference"""
import numpy as np
from scipy.interpolate import pchip_interpolate

cimport numpy as np
cimport cython

__version__ = '0.4'
__all__ = ['infer_intensity', 'regression']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] density_hist(double[:] x, double[:] edges):
    """Density histogram.

    Calculates density of elements x in bins defined by edges. Assumes values
    and edges are sorted, and edges[0] < x < edges[-1]. Behavior for unsorted
    values is undefined.

    Arguments
    ---------
    x : double[:]
        Sorted values
    edges : double[:]
        Bin edges

    Returns
    -------
    double[:] : the density histogram
    """
    cdef double[:] density = np.zeros(edges.shape[0] - 1)
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 1
    cdef Py_ssize_t start = i
    cdef double dist

    while i < n:
        start = i

        while x[i] > edges[j]:
            j = j + 1

        while i < n and x[i] <= edges[j]:
            i = i + 1

        dist = (edges[j] - edges[j - 1])

        if dist != 0.0:
            density[j-1] = (i - start) / dist

    return density


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] map_histogram(double[:] x,
                             double[:] y,
                             double[:] xp,
                             double[:] yp):
    """Map histogram from source (`x`,`y`) to target(`xp`, `yp`). Overwrites `yp`.

    Uses previous-neighbor interpolation. `x` and `xp` are the edges of the
    histogram bins, and `y` are the values of the source histogram bins. `yp`
    gets filled with the mapped values.

    This is used to map a source histogram with variable bin sizes onto a
    target histogram with regular bin sizes. The value `yp[i]` gets the value
    of `y[j]` where `x[i]` is the closest previous point to `xp[j]`. So a large
    bin in the source histogram gets chopped up into a sequence of smaller bins
    in the target histogram, with the all of the target bin values set to the
    source bin value. This works because the bin values represent densities.

    Assumptions:
    * xp[0] == x[0]
    * xp[-1] == x[-1]
    * len(y) == len(x) -1
    * len(yp) == len(xp) -1
    * The values in xp are spaced at least as close as any values of x.
    * `x` and `xp` are sorted.

    For efficiency, these assumptions are not checked. Behavior is undefined if
    they are violated.

Operates in-place on `yp`.

    Arguments
    ---------
    x : double[:]
        Bin edges of source histogram.
    y : double[:]
        Bin values of source histogram.
    xp : double[:]
        Bin edges of target histogram.
    yp : double[:]
        Bin values of target histogram (preallocated, gets overwritten).

    Returns
    -------
    double[:] : yp, filled in-place
    """
    
    cdef Py_ssize_t n = yp.shape[0]
    cdef Py_ssize_t m = y.shape[0]
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t i = 0

    # TODO: If xp[0] == x[0], this should never run. Delete and check that
    # nothing breaks
    while j < n and xp[j] < x[i]:
        yp[j] = 0
        j += 1

    while j < n and i < m:
        while i < m-1 and xp[j] >= x[i+1]:
            i += 1
        yp[j] = y[i]
        j += 1

    return yp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] update_mean(double[:] mean,
                           double[:] vals,
                           Py_ssize_t loop_iter):
    cdef Py_ssize_t N = mean.shape[0]
    cdef Py_ssize_t i = 0
    cdef double m = 0.0
    cdef double v = 0.0
    cdef double x = 0.0
    for i in range(N):
        m = mean[i]
        v = vals[i]
        x = m + ((v - m) / loop_iter)
        mean[i] = x
    return mean


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] get_sequence_boundaries(int num_bins, int num_events, int min_count):
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

    Arguments
    ---------
    n_bins : int
        The number of bins to be defined by the boundaries. Must satisfy
        n_bins <= ((n_events + 2) / min_count) or undefined behavior
        results.
    n_events : int
        The positive number of events to be binned (not counting the overall
        start and end boundaries as events). Must be positive (and nonzero) or
        undefined behavior results.
    min_count : int
        The minimum number of indices between boundaries.

    Returns
    -------
    np.ndarray : Bin boundaries
    """
    if num_bins == 1:
        return np.array([0.0, num_events + 1])

    # The formula here is a simplification of how to get the top end of our
    # noise distribution, which is basically:
    #       end - pad - max(base)
    # Where
    #       end = num_events + 1
    #       pad = min_count - 1
    # and
    #       max(base) == pad + (min_count * (num_bins - 2))
    # for
    #       base = np.arange(pad, min_count * (num_bins - 1), min_count)
    cdef float high = 3 + num_events - (min_count * num_bins)
    cdef Py_ssize_t N = num_bins - 1
    cdef double[:] inner = np.sort(np.random.uniform(low=0, high=high, size=N))

    cdef Py_ssize_t i = 0
    cdef long j = min_count - 1
    for i in range(N):
        inner[i] += j
        j += min_count

    cdef double[:] bounds = np.empty(num_bins + 1)
    bounds[0] = 0.0
    bounds[num_bins] = num_events + 1
    bounds[1:num_bins] = inner
    return bounds


def infer_intensity(events,
                    grid,
                    Py_ssize_t iterations = 100,
                    int min_count = 3):
    """Estimates (potentially nonstationary) event intensity vs. time.

    This class uses Completely Random Average Shifted Histograms (CRASH) to
    compute a continuous curve of event intensity vs. time, as described in **
    Citation TBD **.

    Each histogram is defined by a random number of bin edges, with the
    location of each bin edge sampled uniformly at random between event
    *indices* (not their locations). For example, with the sequence of events
    [1, 2, 3, 100], there is the same probability that an edge will appear
    between 2 and 3 as between 3 and 100.  This allows for the final density
    estimation to adapt its bandwidth to the nonstationarity of event
    locations. A constraint on the minimum number of events per bin keeps
    density peaks from forming pathologically around each event and at
    endpoints.

    Arguments
    ---------
    events : np.ndarray[np.double_t, ndim=1]
        Sorted event times
    grid : np.ndarray[np.double_t, ndim=1]
        Timepoints at which the intensity curve is computed
    iterations : int
        The number of histograms to compute
    min_count : int
        The minimum number of points per bin

    Returns
    -------
    np.ndarray[np.double_t, ndim=1] : Inferred intensity curve
    """
    events = np.asarray(events, dtype=np.double)
    events = events[(grid[0] <= events) & (events <= grid[-1])]

    if iterations < 1:
        raise ValueError('Iteration num must be positive')

    cdef Py_ssize_t n_evts = len(events)
    cdef Py_ssize_t n_grid = len(grid)
    cdef Py_ssize_t max_bins = (n_evts + 1) // min_count
    cdef Py_ssize_t i = 0
    cdef int num_bins = 0

    cdef double[:] mean = np.zeros(n_grid - 1)
    cdef double[:] vals = np.empty(n_grid - 1)

    # np.interpolate args xp and fp are precomputed for efficiency
    cdef np.ndarray xp = np.arange(0, n_evts + 2)
    # This is about 4-5x faster than using np.concatenate with lists
    cdef np.ndarray fp = np.empty(n_evts + 2)
    fp[0] = grid[0]
    fp[-1] = grid[-1]
    fp[1:-1] = events

    # Uninitialized declarations for loop body vars
    cdef double[:] sequence_boundaries, boundaries, h

    for i in range(iterations):
        num_bins = 1 if max_bins < 2 else np.random.randint(1, max_bins + 1)
        # Boundaries are sampled uniformly at random in sequence space, with the
        # constraint that all bins have at least min_count events in them
        # (with endpoints considered events). This means, that a boundary is
        # equally likely to occur between any two events, regardless of the
        # spacing of those events, so long as min_count is respected. This
        # tends to give a smoothness to the final density estimation that varies
        # appropriately with the density of events.
        sequence_boundaries = get_sequence_boundaries(num_bins,
                                                      num_events=n_evts,
                                                      min_count=min_count)
        boundaries = np.interp(sequence_boundaries, xp, fp)
        h = density_hist(events, boundaries)
        # TODO: Fix the fact that if the resolution of `grid` is larger than
        # the minimum spacing between events (1 day, the way we use it), then
        # the mapping can fail because there may be two elements of
        # `boundaries` that fall between the same two grid points, which will
        # mess up the mapping. The fix might be to match the elements of
        # `boundaries` to the nearest `grid` element.
        vals = map_histogram(boundaries, h, grid, vals)
        mean = update_mean(mean, vals, i+1)
    return np.asarray(mean)


def regression(events, values, grid):
    """Estimates values over time.

    Arguments
    ---------
    events : np.ndarray[np.double_t, ndim=1]
        Event times in units of days since an arbitrary reference point
    values : np.ndarray[np.double_t, ndim=1]
        Values for each event time
    grid : np.ndarray[np.double_t, ndim=1]
        Evenly spaced timepoints at which the curve is computed

    Returns
    -------
    array-like of reals : the computed regression curve
    """
    if len(events) != len(values):
        raise ValueError("Events and values are different lengths.")

    if len(events) == 0:
        raise ValueError("Events and values are empty.")

    events = np.asarray(events, dtype=np.double)
    values = np.asarray(values, dtype=np.double)

    mask = (grid[0] <= events) & (events <= grid[-1])
    events = events[mask]
    values = values[mask]

    if len(events) == 1:
        return np.ones(len(grid)) * values[0]

    if len(grid) > 1:
        f_event, f_value = ([grid[0]], [values[0]]
                            ) if grid[0] != events[0] else ([], [])
        l_event, l_value = ([grid[-1]], [values[-1]]
                            ) if grid[-1] != events[-1] else ([], [])
        events = np.concatenate((f_event, events, l_event))
        values = np.concatenate((f_value, values, l_value))

    return pchip_interpolate(events, values, grid)
