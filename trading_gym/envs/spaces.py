from typing import List, Tuple
import numpy as np
from gym import Space
import pandas as pd


class PortfolioVector(Space):
    """
    Data structure for portfolio vector.
    """

    def __init__(self, num_instruments: int, universe: List[str]):

        self.low = np.zeros(num_instruments, dtype=float)

        self.high = np.ones(num_instruments, dtype=float)  # * np.inf

        self.universe = universe

    def sample(self) -> pd.Series:
        """
        Get a random sample of a portfolio vector.
        """
        _vec = np.random.uniform(0.0, 1.0, self.shape[0])
        _vec = _vec / np.sum(_vec)
        _vec = pd.Series(_vec, index=self.universe)
        return _vec

    def contains(self, x: np.array, tolerance: float = 1e-5) -> bool:
        """
        Checks if a vector x is in the space.
        """
        shape_predicate = x.shape == self.shape
        range_predicate = (x >= self.low).all() and (x <= self.high).all()
        budget_constraint = np.abs(x.sum() - 1.0) < tolerance
        return shape_predicate and range_predicate and budget_constraint

    @property
    def shape(self) -> Tuple[int]:
        return self.low.shape

    def __repr__(self) -> str:
        return f"PortfolioVector {self.shape}"

    def __eq__(self, other) -> bool:
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)
