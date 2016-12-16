import numpy as np
cimport numpy as np

def stair_step(x, y, xp, yp):
    """
    Previous neighbor interpolation.

    Args:
        x (list or np.array of numbers): sample points
        y (list or np.array of numbers): sample values (same size as x)
        xp (list or np.array of numbers): query points
        yp (list or np.array of numbers): preallocated list or np.array for
            query values (same size as xp)

    Returns:
        np.array of interpolated values (float)
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
