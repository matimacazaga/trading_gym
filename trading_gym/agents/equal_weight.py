from typing import Deque, Dict
from numpy.core.fromnumeric import shape
from ..envs.spaces import PortfolioVector
from .base import Agent
import numpy as np
from collections import deque
import pandas as pd


class EqualWeightAgent(Agent):

    _id = "ew"

    def __init__(self, action_space: PortfolioVector, window=50, **kwargs):

        self.universe_length = action_space.shape[0]
        self.w = np.full(
            shape=self.universe_length, fill_value=1.0 / self.universe_length
        )

        self.memory = deque(maxlen=window)

    def observe(self, observation: Dict[str, pd.DataFrame], *args, **kwargs):

        self.memory.append(observation["returns"].values)

    def act(self, observation: Dict[str, pd.DataFrame]) -> np.ndarray:

        self.w = np.full(
            shape=self.universe_length, fill_value=1.0 / self.universe_length
        )
        return self.w
