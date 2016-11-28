import numpy as np
cimport numpy as np

def stair_step(x, y, xp, yp):
    cdef int n = len(xp)
    cdef int m = len(y)
    cdef int j = 0
    cdef int i = 0

    while xp[j] < x[i] and j < n:
        yp[j] = 0
        j += 1

    while j < n and i < m:
        while i < m-1 and xp[j] >= x[i+1]:
            i += 1
        yp[j] = y[i]
        j += 1

    return yp
