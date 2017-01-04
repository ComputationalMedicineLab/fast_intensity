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

        fi = FastIntensity(events, start_event=0, end_event=35)
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

    def __init__(self, events, start_event, end_event):
        """
        Initialize an instance.

        Args:
            events (array-like of number): event times in units of days since an
                arbitrary reference point
            start_event (number): beginning of a time range
            end_event (number): end of a time range
        """
        # Convert to numpy array
        events = np.array(events)
        # Cut out of bounds events
        events = np.delete(events, np.where(events < start_event))
        events = np.delete(events, np.where(events > end_event))

        self.events = events
        self.start = start_event
        self.end = end_event

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

        return cls(events, start_e, end_e)

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

        return cls(events, start_e, end_e)

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
        a = self.start
        b = self.end

        nx = int(np.round((b - a) / resolution))

        grid = np.linspace(a, b, nx)
        meanvals = np.zeros(nx)

        vals = np.zeros(len(grid))
        n = len(self.events) + 1
        num_edges = int(2 + np.max([2, np.ceil(density * n)]))

        for i in range(iterations):
            edges = self.randomize_edges(num_edges, density, resolution)
            h = fast_hist(self.events, edges)
            vals = stair_step(edges, h, grid, vals)
            meanvals = meanvals + (vals - meanvals)/(i+1)

        return meanvals

    def randomize_edges(self, num_edges, density, resolution):
        """
        Randomize bin edges for histogram.

        Returns:
            np.array of new bin edges
        """
        n = len(self.events) + 1
        w = n * npr.rand(int(np.max([2, np.ceil(density * n)])))
        w.sort()

        y = np.zeros(num_edges)

        y[0] = self.start
        y[-1] = self.end

        events_w_endpoints = np.concatenate(([self.start], self.events,
                                             [self.end]))

        y[1:-1] = np.interp(w, np.linspace(0, n, n+1), events_w_endpoints)

        y = np.array(y)
        y = np.round(y/resolution)*resolution
        return y
