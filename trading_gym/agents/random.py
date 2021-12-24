from typing import Deque, Dict
from numpy.core.fromnumeric import shape
from ..envs.spaces import PortfolioVector
from .base import Agent
import numpy as np
from collections import deque
import pandas as pd


class RandomAgent(Agent):

    _id = "ew"

    def __init__(self, action_space: PortfolioVector, window=50, **kwargs):

        self.universe_length = action_space.shape[0]
        self.w = np.random.uniform(0.0, 1.0, size=self.universe_length)
        self.w /= self.w.sum()
        self.memory = deque(maxlen=window)

    def observe(self, observation: Dict[str, pd.DataFrame], *args, **kwargs):

        self.memory.append(observation["returns"].values)

    def act(self, observation: Dict[str, pd.DataFrame]) -> np.ndarray:

        self.w = np.random.uniform(0.0, 1.0, size=self.universe_length)
        self.w /= self.w.sum()
        self.w = pd.Series(
            self.w,
            index=observation["returns"].index,
            name=observation["returns"].name,
        )
        return self.w
