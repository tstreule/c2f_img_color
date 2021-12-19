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
        self.buffer: list[tuple[float, int]] = []

    def update(self, val, count=1):
        self.counter += count
        self.buffer += [(val, count)]

        delta1 = val - self._M
        self._M += count * delta1 / self.counter
        delta2 = val - self._M
        self._S += count * delta1 * delta2

    @property
    def mean(self):
        return self._M

    @property
    def std(self):
        if self.counter == 1:
            return 0.0
        return np.sqrt(self._S / self.counter)
