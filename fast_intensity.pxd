# Cython header files like this enable us to cimport cdef and cpdef functions
# from our pyx code (which allows us to test c only funcs)

# cython: language_level=3

cdef double[:] density_hist(double[:] x, double[:] edges)
cdef double[:] stair_step(double[:] x, double[:] y, double[:] xp, double[:] yp)
cdef double[:] get_sequence_boundaries(int num_bins, int num_events, int min_count)
