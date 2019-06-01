#!/usr/bin/env ipython
"""Benchmark fast intensity

Loads some test data, runs a benchmark using ipython magic, and then drops into
an IPython interactive shell.
"""
import IPython
import numpy

import pyximport
pyximport.install()

from fast_intensity import *

if __name__ == '__main__':
    ipython = IPython.get_ipython()

    # Regression cannot be accelerated at this time, no point benching
    #reg_sample = numpy.load('regression.npz')
    #events = reg_sample['events']
    #values = reg_sample['values']
    #grid = reg_sample['grid']
    #stmt = 'fast_regression(events, values, grid)'
    #reg_result = ipython.run_cell_magic('timeit', '-o', stmt)

    setup = 'fi = FastIntensity(evts, grid)'
    stmt = 'fi.run_inference()'

    inf_sample = numpy.load('intensity.npz')
    evts = inf_sample['events']
    grid = inf_sample['grid']
    ipython.run_cell_magic('timeit', setup, stmt)

    # Run a bigger one 10_000 events over 20 years, sampled daily
    n = 20 * 365
    numpy.random.seed(42)
    days = numpy.arange(0.0, n)
    numpy.random.shuffle(days)
    evts = numpy.sort(days[:10_000])
    grid = numpy.linspace(1.0, n, n+1)
    ipython.run_cell_magic('timeit', setup, stmt)
