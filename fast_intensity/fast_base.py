# Copyright 2017 Thomas A. Lasko, Jacek Bajor


class FastBase(object):
    def __init__(self, grid):
        """Initialize with parameters.

        Args:
            grid (array-like of reals): Evenly spaced time points at which to
                compute the curve.
        """
        self.grid = grid

    @property
    def start(self):
        return self.grid[0]

    @property
    def end(self):
        return self.grid[-1]

    @property
    def resolution(self):
        return self.grid[1] - self.grid[0]

    def run_inference(self):
        raise NotImplementedError
