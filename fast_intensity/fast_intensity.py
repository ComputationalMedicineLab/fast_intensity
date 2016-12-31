from datetime import datetime

import numpy as np
import numpy.random as npr

# Import and compile Cython files
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from .stair_step import stair_step
from .fast_hist import fast_hist

class FastIntensity(object):
    """Estimates (potentially nonstationary) event intensity vs. time.

    This class uses Completely Random Average Shifted Histograms (CRASH) to
    compute a continuous curve of event  intensity vs. time, as described in **
    Citation TBD **.

    Each histogram is defined by a random number of bin edges, with the location
    of each bin edge  sampled uniformly at random between event *indices* (not
    their locations). For example, with the sequence of events  [1, 2, 100],
    there is the same probability that an edge will appear between 1 and 2 as
    between 2 and 100.  This allows for the final density estimation to adapt
    its 'bandwidth' to the nonstationarity of event locations.  The smoothness
    and resolution of the resulting curves can be set.

    Usage:
        events = [10, 15, 16, 17, 28]
        events_with_endpoints = [-1] + events + [35]

        fi = FastIntensity(events_with_endpoints)
        intensity = fi.run_inference()

        fi = FastIntensity.from_events(events, start_event=-1, end_event=35)
        intensity = fi.run_inference()

        dates = [dt.datetime(2000, 1, 2), dt.datetime(2000, 1, 10),
                 dt.datetime(2000, 1, 15), dt.datetime(2000, 2, 1)]

        fi = FastIntensity.from_dates(dates, start_date=dt.datetime(2000, 1, 1),
                                      end_date=dt.datetime(2000, 3, 1))
        intensity = fi.run_inference()

        date_strs = ['2000-01-02', '2000-01-10', '2000-01-15', '2000-02-01']

        fi = FastIntensity.from_string_dates(date_strs, start_date='2000-01-01',
                                             end_date='2000-03-01',
                                             date_format='%Y-%m-%d')
        intensity = fi.run_inference()
    """

    def __init__(self, events_with_endpoints):
        """
        Initialize with endpoints included as the first and last element of
        events_with_endpoints.

        This is the most direct, but least convenient or intuitive
        initialization.

        Args:
            events_with_endpoints (array-like of numbers):
                event times in units of days since an arbitrary reference point
                (often the first event).  The first and last elements define the
                time range over which the curves are computed,  and are not
                counted as events themselves.
        """
        self.events_with_endpoints = events_with_endpoints

    @classmethod
    def from_events(cls, events, start_event, end_event):
        """
        Add endpoints and initialize an instance.

        Args:
            events (array-like of number)
            start_event (number)
            end_event (number)
        """
        # Convert to numpy array
        events = np.array(events)
        # Cut out of bounds events
        events = np.delete(events, np.where(events <= start_event))
        events = np.delete(events, np.where(events >= end_event))

        events_with_endpoints = np.zeros(len(events) + 2)
        events_with_endpoints[0] = start_event
        events_with_endpoints[-1] = end_event
        events_with_endpoints[1:-1] = events
        return cls(events_with_endpoints)

    @classmethod
    def from_dates(cls, dates, start_date, end_date):
        """
        Convert dates/datetimes to events and initialize an instance.

        Args:
            dates (array-like of date/datetime)
            start_event (date/datetime)
            end_event (date/datetime)
        """
        events, start_e, end_e = FastIntensity.convert_dates_to_events(dates,
                                    start_date, end_date)

        return cls.from_events(events, start_e, end_e)

    @classmethod
    def from_string_dates(cls, dates, start_date, end_date,
                          date_format='%Y-%m-%d %H:%M:%S'):
        """
        Convert date strings to events and initialize an instance.

        Args:
            dates (array-like of strings): dates represented by correctly
                formatted strings
            start_event (string): date represented by a correctly formatted
                string
            end_event (string): date represented by a correctly formatted
                string
            date_format (string): format of dates in the input (same as used
                in datetime.datetime.strptime() function)
        """
        start_date = datetime.strptime(start_date, date_format)
        end_date = datetime.strptime(end_date, date_format)
        dates = [ datetime.strptime(d, date_format) for d in dates]
        events, start_e, end_e = FastIntensity.convert_dates_to_events(dates,
                                    start_date, end_date)

        return cls.from_events(events, start_e, end_e)

    @staticmethod
    def time_delta_in_days(a, b):
        """
        Return time difference in days.

        Args:
            a, b (date or datetime)

        Returns:
            float: time difference in days (exact, not rounded)
        """
        return (a-b).total_seconds()/(24*60*60)

    @staticmethod
    def convert_dates_to_events(dates, start_date, end_date):
        """
        Convert dates to events.

        Args:
            dates (array-like of date/datetime)
            start_event (date/datetime)

        Returns:
            list of numbers representing events
        """
        events = np.zeros(len(dates))
        for i, d in enumerate(dates):
            events[i] = FastIntensity.time_delta_in_days(d, start_date)
        return events, 0, FastIntensity.time_delta_in_days(end_date, start_date)

    def run_inference(self, density=0.00274, resolution=1, iterations=100):
        """
        Run event intensity inference.

        Args:
            density (number): average number of bin edges between neighboring
                points (default 1/365).
            resolution (number): resolution for bin edges in units of days
                (default 1).
            iterations (int): number of inference iterations (default: 100)

        Returns
            np.array of event intensity
        """
        a = self.events_with_endpoints[0]
        b = self.events_with_endpoints[-1]
        events = self.events_with_endpoints[1:-1]

        nx = int(np.round((b - a) / resolution))
        a = np.floor(b - nx * resolution)

        grid = np.linspace(a, b, nx)
        meanvals = np.zeros(nx)

        vals = np.zeros(len(grid))
        n = len(self.events_with_endpoints) - 1
        num_edges = int(2 + np.max([2, np.ceil(density * n)]))

        edges = np.zeros(num_edges)
        edges[0] = a
        edges[-1] = b

        for i in range(iterations):
            edges = self.randomize_edges(self.events_with_endpoints, edges,
                density, resolution)
            h = fast_hist(events, edges)
            vals = stair_step(edges, h, grid, vals)
            meanvals = meanvals + (vals - meanvals)/(i+1)

        return meanvals

    def randomize_edges(self, x, y, density, resolution):
        """
        Randomize bin edges for histogram.

        Args:
            x (array-like of numbers): collection of events
            y (array-like of numbers): previous edges

        Returns:
            np.array of new bin edges
        """
        n = len(x) - 1
        w = n * npr.rand(int(np.max([2, np.ceil(density * n)])))
        w.sort()

        y[0] = x[0]
        y[-1] = x[-1]

        y[1:-1] = np.interp(w, np.linspace(0, n, n+1), x)

        y = np.array(y)
        y = np.round(y/resolution)*resolution
        return y
