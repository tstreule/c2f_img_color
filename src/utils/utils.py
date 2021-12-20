import numpy as np


__all__ = ["WelfordMeter"]


class WelfordMeter:
    """
    Implements Welford's online algorithm for running averages and standard deviations.

    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    - https://www.kite.com/python/answers/how-to-find-a-running-standard-deviation-in-python
    """

    def __init__(self):
        self.counter = 0
        self._M = 0
        self._S = 0
        self.buffer: list[tuple[float, float]] = []

    def update(self, val, count=1):
        self.counter += count

        delta1 = val - self._M
        self._M += count * delta1 / self.counter
        delta2 = val - self._M
        self._S += count * delta1 * delta2

        self.buffer += [(self.mean, self.std)]

    def reset(self):  # keep buffer as is but reset running averages and stds
        self.counter = 0
        self._M = 0
        self._S = 0

    @property
    def mean(self):
        return self._M

    @property
    def std(self):
        if self.counter == 1:
            return 0.0
        return np.sqrt(self._S / self.counter)
