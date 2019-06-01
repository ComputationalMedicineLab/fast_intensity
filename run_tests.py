#!/usr/bin/env python
import tempfile
import unittest

import numpy
import pyximport
# Since we're running a test suite, we don't want the built files to be cached
# but to be rebuilt on every run, so we tell pyximport to use a temporary
# directory as the build cache
with tempfile.TemporaryDirectory() as tmpdir:
    pyximport.install(build_dir=tmpdir)
    from test_fast_intensity import *
    unittest.main()
