import numpy as np
cimport numpy as np

def fast_hist(x, edges):
    # fast_hist Returns density of elements x in bins defined by edges.
    # Assumes x and edges are sorted, and edges(1) < x < edges(end)
    cdef np.ndarray density = np.zeros(len(edges) - 1, dtype=np.float)
    cdef int n = len(x)
    cdef int i = 0
    cdef int j = 1
    cdef int i_start = i

    while i < n:
        i_start = i

        while x[i] > edges[j]:
            j = j + 1

        while i < n and x[i] <= edges[j]:
            i = i + 1

        density[j-1] = (i - i_start) / (edges[j] - edges[j - 1])

    return density
