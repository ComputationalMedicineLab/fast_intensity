# Copyright 2017 Thomas A. Lasko, Jacek Bajor

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

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
