#!/usr/bin/env ipython -i
"""Benchmark fast intensity

Loads some test data, runs a benchmark using ipython magic, and then drops into
an IPython interactive shell.
"""
import IPython
import numpy
import pyximport
pyximport.install(setup_args={"include_dirs": numpy.get_include()})
from fast_intensity import *

if __name__ == '__main__':
    ipython = IPython.get_ipython()
    reg_sample = numpy.load('regression.npz')
    inf_sample = numpy.load('intensity.npz')

    events = reg_sample['events']
    values = reg_sample['values']
    grid = reg_sample['grid']
    setup = f'-o reg = FastRegression(events, values, grid)'
    stmt = 'reg.run_inference()'
    reg_result = ipython.run_cell_magic('timeit', setup, stmt)

    events = inf_sample['events']
    grid = inf_sample['grid']
    setup = f'-o inf = FastIntensity(events, grid)'
    stmt = 'inf.run_inference()'
    inf_result = ipython.run_cell_magic('timeit', setup, stmt)
