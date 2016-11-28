import numpy as np
import numpy.random as npr

# Import and compile Cython files
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from fast_intensity.stair_step import stair_step
from fast_intensity.fast_hist import fast_hist

# events = [-1,1937,1939,1945,1979,1986,2026,2029,2189,2211,2212,2213,2214,2216,
#           2226,2238,2240,2243,2247,2279,2364,2369,2380,2408,2412,2413,2420,2450,
#           2457,2541,2590,2642,2672,2701,2803,2971,3010,3153,3295,3336,3395,3625,
#           3659,3723,3766,3876,4011,4240]
#
# fast_density_inference(events, 1, 1.0/365, 1)


def fast_density_inference(events_with_endpoints, n_pool=1, density=1, resolution=1):
    a = events_with_endpoints[0]
    b = events_with_endpoints[-1]
    events = events_with_endpoints[1:-1]

    nx = int(np.round((b - a) / resolution))
    a = np.floor(b - nx * resolution)

    grid = np.linspace(a, b, nx)
    meanvals = np.zeros(nx)
    num_iter = 100

    vals = np.zeros(len(grid))
    n = len(events_with_endpoints) - 1
    num_edges = int(2 + np.max([2, np.ceil(density * n)]))

    edges = np.zeros(num_edges)
    edges[0] = a
    edges[-1] = b

    for i in range(num_iter):
        edges = draw_points(events_with_endpoints, resolution, edges, density)
        h = fast_hist(events, edges)
        # h = np.histogram(events, edges, density=True)[0]
        vals = stair_step(edges, h, grid, vals)
        meanvals = meanvals + (vals - meanvals)/(i+1)

    return meanvals*len(events_with_endpoints)


def draw_points(x, res, y, density=1):
    a = x[0]
    b = x[-1]

    n = len(x) - 1 # edge numbers, including endpoints
    w = n * npr.rand(int(np.max([2, np.ceil(density * n)])))
    w.sort()

    y[0] = a
    y[-1] = b

    y[1:-1] = np.interp(w, np.linspace(0,n,n+1), x)

    y = np.array(y)
    y = np.round(y/res)*res
    return y
