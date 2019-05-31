#!/usr/bin/env ipython -i
"""Benchmark fast intensity

Loads some test data, runs a benchmark using ipython magic, and then drops into
an IPython interactive shell.
"""
import argparse
import pathlib

import IPython
import numpy
import pyximport
pyximport.install(setup_args={"include_dirs": numpy.get_include()})
from fast_intensity import *

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('datfile', type=pathlib.Path)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--number', type=int, default=10)
    args = p.parse_args()

    data = numpy.load(args.datfile)
    events = data['events']
    values = data['values']
    grid = data['grid']

    ipython = IPython.get_ipython()
    magic = 'timeit'
    setup = (f'-o -r {args.repeat} -n {args.number} '
              'reg = FastRegression(events, values, grid)')
    stmt = 'reg.run_inference()'

    result = ipython.run_cell_magic(magic, setup, stmt)
