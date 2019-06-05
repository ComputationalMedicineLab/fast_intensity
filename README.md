fast-intensity
===============================

authors: Thomas A. Lasko, Jacek Bajor, John Still


Overview
--------
Fast density inference. Generates intensity curves from given events.


Installation
------------
We only support Python 3 and above.  We recommend using `fast_intensity` in a
[virtual environment](https://docs.python.org/3/tutorial/venv.html); however,
if you choose to install to a system-wide version of Python, be aware that some
distributions will alias Python 3's `pip` as `pip3`.  You should be able to
verify which Python `pip` is associated with by running `pip --version`.

If you prefer to install a precompiled binary, we provide wheels for OS X and
Linux (via the [manylinux](https://github.com/pypa/manylinux) project).  The
basic pip install command line

    $ pip install fast-intensity

should prefer one of our prebuilt binaries. Installation from source requires
an environment with Cython, numpy, and scipy preinstalled.

    $ pip install cython numpy scipy

Then you may install a release from source by specifying _not_ to use a binary:

    $ pip install fast-intensity --no-binary fast-intensity

(Yes, it is necessary to specify `fast-intensity` twice.)  Alternately, to
install the bleeding edge version:

    $ git clone https://github.com/ComputationalMedicineLab/fast-intensity.git
    $ cd fast-intensity
    $ pip install -e .


Usage
-----
```python
%matplotlib inline
%load_ext cython
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
# If working locally rather than from a wheel or other binary
import pyximport
pyximport.install(language_level=3)
from fast_intensity import infer_intensity

np.random.seed(42)

# Specify a series of 100 events spread over a year
days = np.arange(0.0, 365)
np.random.shuffle(days)
events = np.sort(days[:100])

# Specify times (as reals) where we want to calculate the intensity of event occurrence
grid = np.linspace(1, 365, num=12)

# Generate the intensity curve - the unit is events per time unit
curve = infer_intensity(events, grid)
print(curve)
#     array([0.38953   , 0.27764734, 0.33549508, 0.27285165, 0.22284481, 0.16997545,
#            0.26651725, 0.23580527, 0.23351076, 0.25272662, 0.33146159, 0.28486727])

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(events, np.zeros(len(events)), alpha='0.4', label='Events')
ax.scatter(grid, np.zeros(len(grid)) + 0.025, label='Grid')
ax.plot(grid, curve, label='Intensity')
plt.legend()
plt.show()
```

You can see how the intensity graph dips in the middle, where events are more thinly spaced,
and rises near the beginning (where we have a high density of events).

![figure](https://github.com/ComputationalMedicineLab/fast_intensity/blob/master/intensity_figure.png)
