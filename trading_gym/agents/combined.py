from .base import Agent
from ..envs.spaces import PortfolioVector
from typing import Optional, Dict
from ..utils.screener import Screener
import pandas as pd
from collections import deque
import numpy as np


class CombinedAgent(Agent):
    def __init__(
        self,
        action_space: PortfolioVector,
        window: int = 50,
        rebalance_each_n_obs: int = 7,
        agent_name: str = "dlpopt",
    ):
        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory_rewards_1 = deque(maxlen=window)
        self.memory_rewards_2 = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0
        self._id = agent_name + "_combined"

    def observe(self, reward_1, reward_2, *args, **kwargs) -> None:

        self.memory_rewards_1.append(reward_1)
        self.memory_rewards_2.append(reward_2)

    def act(self, observation, action_1, action_2) -> pd.Series:
        if (
            self.rebalance_counter % self.rebalance_each_n_obs == 0
            or self.observation_size != observation["returns"].shape[0]
        ):
            ws = 1.0 + np.array(
                [
                    np.array(self.memory_rewards_1).mean(),
                    np.array(self.memory_rewards_2).mean(),
                ]
            )

            ws = ws / ws.sum()

            N_1 = ws[0] / action_1.sum()

            w_1 = pd.Series(N_1 * action_1, index=action_1.index, name=action_1.name)

            N_2 = ws[1] / action_2.sum()

            w_2 = pd.Series(N_2 * action_2, index=action_2.index, name=action_2.name)

            w = w_1.copy()

            for ind in w_2.index:
                if ind in w.index:
                    w[ind] += w_2[ind]
                else:
                    w[ind] = w_2[ind]

            self.w = w

        self.rebalance_counter += 1

        return self.w
