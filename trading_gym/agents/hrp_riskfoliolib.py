from typing import Dict, Optional
import numpy as np
import pandas as pd
from ..utils.screener import Screener
from .base import Agent
from collections import deque
from ..envs.spaces import PortfolioVector
from riskfolio import HCPortfolio


class HRPAgent(Agent):

    _id = "hrp_riskfolio"

    def __init__(
        self,
        action_space: PortfolioVector,
        window: int = 50,
        screener: Optional[Screener] = None,
        model: str = "HRP",
        objective: str = "MinRisk",
        codependence: str = "pearson",
        risk_measure: str = "MV",
        covariance: str = "hist",
        rf: float = 0.0,
        max_k: int = 10,
        leaf_order: bool = True,
        rebalance_each_n_obs: int = 1,
        w_min: float = 0.0,
        w_max: float = 1.0,
        volatility_scaling: bool = False,
        target_volatility: Optional[float] = None,
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory_returns = deque(maxlen=window)
        self.memory_volume = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.screener = screener
        self.model = model
        self.objective = objective
        self.codependence = codependence
        self.risk_measure = risk_measure
        self.covariance = covariance
        self.rf = rf
        self.max_k = max_k
        self.leaf_order = leaf_order
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0
        self.w_min = w_min
        self.w_max = w_max
        self.volatility_scaling = volatility_scaling
        self.target_volatility = target_volatility

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs):

        self.memory_returns.append(observation["returns"])
        self.memory_volume.append(observation["volume"])

    def act(self, observation: Dict[str, pd.Series]) -> np.ndarray:

        if len(self.memory_returns) != self.memory_returns.maxlen:
            self.w = self.action_space.sample()
            return self.w

        returns = pd.DataFrame(self.memory_returns).dropna(axis=1)

        if self.volatility_scaling:
            volatility = returns.std(axis=0, ddof=1)
            returns = returns * (self.target_volatility / volatility)

        volume = pd.DataFrame(self.memory_volume).loc[:, returns.columns]

        if self.screener:
            if isinstance(self.screener, tuple) or isinstance(self.screener, list):
                for screener in self.screener:
                    assets_list = screener.filter(returns, volume)
                    returns = returns.loc[:, assets_list]
                    volume = volume.loc[:, assets_list]
            else:
                assets_list = self.screener.filter(returns, volume)
                returns = returns.loc[:, assets_list]

        if (
            self.rebalance_counter % self.rebalance_each_n_obs == 0
            or self.observation_size != returns.shape[1]
        ):

            self.observation_size = returns.shape[1]

            w_min_series = pd.Series(
                np.full(returns.shape[1], self.w_min), index=returns.columns
            )

            w_max_series = pd.Series(
                np.full(returns.shape[1], self.w_max), index=returns.columns
            )

            hrp_algo = HCPortfolio(returns, w_max=w_max_series, w_min=w_min_series)

            w = hrp_algo.optimization(
                model=self.model,
                obj=self.objective,
                codependence=self.codependence,
                rm=self.risk_measure,
                rf=self.rf,
                covariance=self.covariance,
                linkage="single",
                max_k=self.max_k,
                leaf_order=self.leaf_order,
            )

            w = pd.Series(
                w.loc[:, "weights"],
                index=w.index,
                name=observation["returns"].name,
            )

            if np.any(w < 0):
                w += np.abs(w.min())

            print(w.sum())
            if w.sum() < 1.0 - 1e-3:
                diff = 1.0 - w.sum()
                while diff != 0.0:
                    print(diff)
                    for ind in w.index:
                        if w[ind] + diff / len(w) < w_max_series[ind]:
                            print(ind, w[ind], w[ind] + diff / len(w))
                            w[ind] += diff / len(w.index)
                            diff -= diff / len(w.index)

            if w.sum() > 1.0 + self.tol:
                w /= w.sum()

            self.w = w

        self.rebalance_counter += 1

        return self.w
