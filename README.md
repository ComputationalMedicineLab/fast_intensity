fast-intensity
===============================

version number: 0.1.5

authors: Thomas A. Lasko, Jacek Bajor

Overview
--------

Fast density inference. Generates intensity curves from given events.

Installation
------------

To install use pip:

    $ pip install fast-intensity


Or clone the repo:

    $ git clone https://github.com/ComputationalMedicineLab/fast-intensity.git
    $ python setup.py install


Usage
-----

```python
from fast_intensity import FastIntensity

# Basic usage with events including endpoints
events = [10, 15, 16, 17, 28]
events_with_endpoints = [-1] + events + [35]
fi = FastIntensity(events_with_endpoints)
intensity = fi.run_inference()

# Provided events don't have endpoints. Left and right bounds passed as an argument
fi = FastIntensity.from_events(events, start_event=-1, end_event=35)
intensity = fi.run_inference()

# Events and endpoints as date or datetime object
dates = [dt.datetime(2000, 1, 2), dt.datetime(2000, 1, 10),
         dt.datetime(2000, 1, 15), dt.datetime(2000, 2, 1)]

fi = FastIntensity.from_dates(dates, start_date=dt.datetime(2000, 1, 1),
                              end_date=dt.datetime(2000, 3, 1))
intensity = fi.run_inference()

# Events and endpoints as string representing time or date
date_strings = ['2000-01-02', '2000-01-10', '2000-01-15', '2000-02-01']

fi = FastIntensity.from_string_dates(date_strings, start_date='2000-01-01',
                                     end_date='2000-03-01',
                                     date_format='%Y-%m-%d')
intensity = fi.run_inference()

# Displaying intensity with matplotlib
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, drange

plt.style.use('ggplot')

date_strings = ['2016-04-26','2016-04-27','2016-04-28','2016-04-29','2016-04-30',
  '2016-05-01','2016-05-02','2016-05-03','2016-05-04','2016-05-05','2016-09-01',
  '2016-09-02','2016-09-03','2016-09-04','2016-09-05','2016-09-06','2016-09-07',
  '2016-09-08','2016-09-09','2016-10-09','2016-10-10','2016-12-09', '2016-12-10']
fi = FastIntensity.from_string_dates(date_strings, start_date='2016-01-01',
                                     end_date='2016-12-31',
                                     date_format='%Y-%m-%d')
intensity = fi.run_inference(1000)

months = MonthLocator(range(0, 13), bymonthday=1, interval=1)
monthsFmt = DateFormatter("%b %Y")
days = drange(datetime.date(2015, 12, 31), datetime.date(2017, 1, 1),
              datetime.timedelta(days=1))
fig, ax = plt.subplots()
ax.plot_date(days, intensity, '-')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.autoscale_view()
fig.autofmt_xdate()
plt.show()
```

![figure](https://github.com/ComputationalMedicineLab/fast_intensity/raw/master/intensity_figure.png "Figure")
