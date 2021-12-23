import numpy as np
from gym import Space


class PortfolioVector(Space):
    """
    Data structure for portfolio vector.
    """

    def __init__(self, num_instruments):

        self.low = np.zeros(num_instruments, dtype=float)

        self.high = np.ones(num_instruments, dtype=float)  # * np.inf

    def sample(self):
        """
        Get a random sample of a portfolio vector.
        """
        _vec = np.random.uniform(0.0, 1.0, self.shape[0])
        return _vec / np.sum(_vec)

    def contains(self, x: np.array, tolerance=1e-5):
        """
        Checks if a vector x is in the space.
        """
        shape_predicate = x.shape == self.shape
        range_predicate = (x >= self.low).all() and (x <= self.high).all()
        budget_constraint = np.abs(x.sum() - 1.0) < tolerance
        return shape_predicate and range_predicate and budget_constraint

    @property
    def shape(self):
        return self.low.shape

    def __repr__(self):
        return f"PortfolioVector {self.shape}"

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)
