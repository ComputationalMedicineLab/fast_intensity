import numpy as np
cimport numpy as np

def fast_hist(x, edges):
    """
    Return density histogram.

    Calculates density of elements x in bins defined by edges. Assumes values
    and edges are sorted, and edges[0] < x < edges[-1]

    Args:
        x (array-like of numbers): values
        edges (array-like of numbers): bin edges (2 or more values)

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

        density[j-1] = (i - start) / (edges[j] - edges[j - 1])

    return density
