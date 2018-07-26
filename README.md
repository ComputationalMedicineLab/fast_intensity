fast-intensity
===============================

authors: Thomas A. Lasko, Jacek Bajor

Overview
--------

Fast density inference. Generates intensity curves from given events.

Installation
------------

If you prefer to install a precompiled binary, we provide wheels for OS X and
Linux (via the [manylinux](https://github.com/pypa/manylinux) project).  The
basic pip install command line

    $ pip install fast-intensity

should prefer one of our prebuilt binaries. Installation from source requires
an environment with Cython and numpy preinstalled.  

    $ pip install cython numpy

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
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from fast_intensity import FastIntensity

np.random.seed(42)

# Specify a series of 100 events spread over a year
days = np.arange(0, 365)
np.random.shuffle(days)
events = sorted(days[:100])

# Specify times (as reals) where we want to calculate the intensity of event occurrence
grid = np.linspace(1, 365, num=12)

# Configure a FastIntensity instance with the events and the grid
curve_builder = FastIntensity(events, grid)

# Generate the intensity curve - the unit is events per time unit
intensity = curve_builder.run_inference()
print(intensity)
#     array([0.38953   , 0.27764734, 0.33549508, 0.27285165, 0.22284481,
#            0.16997545, 0.26651725, 0.23580527, 0.23351076, 0.25272662,
#            0.33146159, 0.28486727])

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(events, np.zeros(len(events)), alpha='0.4', label='Events')
ax.scatter(grid, np.zeros(len(grid)) + 0.025, label='Grid')
ax.plot(grid, intensity, label='Intensity')
plt.legend()
plt.show()
```

You can see how the intensity graph dips in the middle, where events are more thinly spaced,
and rises near the beginning (where we have a high density of events).

![figure](https://github.com/ComputationalMedicineLab/fast_intensity/blob/master/intensity_figure.png)
