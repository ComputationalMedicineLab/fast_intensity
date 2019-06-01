# Compiler directives have to come before code, but may come after comments
# cython: language_level=3
"""Fast intensity inference"""
import numpy as np
from scipy.interpolate import pchip_interpolate

cimport numpy as np

__version__ = '0.4.dev0'
__all__ = ['infer_intensity', 'regression']


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
    cdef double[:] density = np.zeros(len(edges) - 1)
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


cdef double[:] stair_step(double[:] x,
                          double[:] y,
                          double[:] xp,
                          double[:] yp):
    """Previous neighbor interpolation.  Behavior undefined for unsorted points.

    Operates in-place on argument `yp`. Behavior undefined for unsorted points.

    Arguments
    ---------
    x : double[:]
        Sorted sample points
    y : double[:]
        Sample values (same size as x)
    xp : double[:]
        Query points
    yp : double[:]
        Preallocated buffer or np.array for query values (same size as xp)

    Returns
    -------
    double[:] : yp, augmented in-place
    """
    cdef int n = len(xp)
    cdef int m = len(y)
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
    # The bin at each end must contain at least pad = min_count - 1 actual
    # events, because the endpoints count as an included event, even though the
    # intervals stop exactly at that event.
    start = 0
    end = num_events + 1
    pad = min_count - 1

    boundaries = np.empty(num_bins + 1)
    boundaries[0] = start
    boundaries[-1] = end
    if num_bins == 1:
        return boundaries

    boundaries[1:-1] = np.arange(start=pad,
                                 stop=min_count * (num_bins - 1),
                                 step=min_count)
    slop = np.random.uniform(low=0,
                             high=end - pad - boundaries[-2],
                             size=num_bins - 1)
    slop.sort()
    np.add(boundaries[1:-1], slop, out=boundaries[1:-1])
    return boundaries


def infer_intensity(events, grid, iterations=100, min_count=3):
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
    before_start = np.where(events < grid[0])
    events = np.delete(events, before_start)

    after_end = np.where(events > grid[-1])
    events = np.delete(events, after_end)

    meanvals = np.zeros(len(grid))
    vals = np.zeros(len(grid))
    n = len(events) + 1

    # Compute event_indices once for all iterations of _get_boundaries, for
    # efficiency. (This has a measurable effect on run time.)
    event_indices = np.linspace(0, n, n + 1)

    events_w_endpoints = np.concatenate(([grid[0]], events, [grid[-1]]))
    max_bins = int(event_indices[-1] // min_count)

    for i in range(iterations):
        if max_bins < 2:
            num_bins = 1
        else:
            num_bins = np.random.randint(1, max_bins + 1)
        # Boundaries are sampled uniformly at random in sequence space, with the
        # constraint that all bins have at least min_count events in them
        # (with endpoints considered events). This means, that a boundary is
        # equally likely to occur between any two events, regardless of the
        # spacing of those events, so long as min_count is respected. This
        # tends to give a smoothness to the final density estimation that varies
        # appropriately with the density of events.
        sequence_boundaries = get_sequence_boundaries(num_bins,
                                                       num_events=len(events),
                                                       min_count=min_count)
        boundaries = np.interp(sequence_boundaries,
                               event_indices,
                               events_w_endpoints)
        h = density_hist(events.astype(np.double), boundaries.astype(np.double))
        vals = stair_step(boundaries, h, grid, vals)
        meanvals = meanvals + (vals - meanvals) / (i + 1)

    return meanvals


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

    events = np.array(events, dtype=np.double)
    values = np.array(values, dtype=np.double)

    # Cut out of bounds values
    before_start = np.where(events < grid[0])
    values = np.delete(values, before_start)
    events = np.delete(events, before_start)

    after_end = np.where(events > grid[-1])
    values = np.delete(values, after_end)
    events = np.delete(events, after_end)

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
