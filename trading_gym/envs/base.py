from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Box
from ..agents.base import Agent
from abc import abstractmethod
from typing import Dict, Optional, List, Tuple, Union
from ..utils.pandas_utils import align_index, clean
from .spaces import PortfolioVector
from ..utils.summary import stats


class BaseEnv(Env):

    """
    Base trading environment based on OpenAI gym.


    """

    class Record:
        """
        Local class for registering actions and rewards.

        Attributes
        ----------
        actions: pd.DataFrame
            Table with actions taken by the agent.
        rewards: pd.DataFrame
            Table with rewards received by the agent.
        """

        def __init__(self, index, columns):

            # self.actions = pd.DataFrame(columns=columns, index=index, dtype=float)
            self.actions = {}

            self.actions[index[0]] = pd.Series(
                {col: 0.0 if col != "Cash" else 1.0 for col in columns}
            )

            self.rewards = {}

            self.rewards[index[0]] = pd.Series({col: 0.0 for col in columns})
            # self.actions.iloc[0] = np.zeros(len(columns))

            # self.actions.iloc[0]["Cash"] = 1.0

            # self.rewards = pd.DataFrame(columns=columns, index=index, dtype=float)

            # self.rewards.iloc[0] = np.zeros(len(columns))

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        assets_data: Optional[Dict[str, pd.DataFrame]] = None,
        returns: Optional[pd.DataFrame] = None,
        cash: bool = True,
        risk_free_rate: float = 0.08,
        fee: float = 0.0,
        **kwargs,
    ):

        if assets_data is not None and isinstance(assets_data, dict):

            self.close = assets_data["close"]

            self.open = assets_data["open"]

            self.high = assets_data["high"]

            self.low = assets_data["low"]

            self.volume = assets_data["volume"]

        elif universe is not None and isinstance(universe, list):

            assets_data = self._get_assets_data(
                kwargs["start"], kwargs["end"], universe
            )

            self.close = assets_data["close"]

            self.open = assets_data["open"]

            self.high = assets_data["high"]

            self.low = assets_data["low"]

            self.volume = assets_data["volume"]

        else:

            raise ValueError("Either universe or assets_data must be provided.")

        if cash:
            self.close.loc[:, "Cash"] = 1.0
            self.open.loc[:, "Cash"] = 1.0
            self.high.loc[:, "Cash"] = 1.0
            self.low.loc[:, "Cash"] = 1.0
            self.volume.loc[:, "Cash"] = 1.0

        if returns is not None and isinstance(returns, pd.DataFrame):
            self._returns = returns
        elif returns is None and (
            self.close is not None and isinstance(self.close, pd.DataFrame)
        ):
            self._returns = self.close.pct_change(fill_method=None).iloc[
                1:
            ]  #! fill_method=None for avoiding filling NAs with 0, which may produce series with zero variance if a coin goes out of market.
        else:
            raise ValueError(
                "Either 'returns' must be not None or 'close' must be a pandas DataFrame."
            )

        if cash:
            self._returns.loc[:, "Cash"] = (
                1.0 + np.random.normal(risk_free_rate, 0.01, size=len(self._returns))
            ) ** (1 / 365.0) - 1

        self.close = align_index(self._returns, self.close)

        self.open = align_index(self._returns, self.open)

        self.high = align_index(self._returns, self.high)

        self.low = align_index(self._returns, self.low)

        self.volume = align_index(self._returns, self.volume)

        num_instruments = len(self.universe)

        self.action_space = PortfolioVector(num_instruments, self.universe)

        self.observation = Box(
            low=-np.inf, high=np.inf, shape=(num_instruments,), dtype=np.float32
        )

        self._counter = 0

        self.agents = {}

        self._pnl = pd.DataFrame(
            index=self.dates, columns=[agent.name for agent in self.agents]
        )

        self._fig, self._axes = None, None

        self.fee = fee

    @property
    def universe(self) -> List[str]:
        """
        Asset list.
        """
        return self.close.columns.tolist()

    @property
    def dates(self) -> pd.Index:
        """
        Dates of the prices in the environment.
        """
        return self.close.index

    @property
    def index(self) -> pd.DatetimeIndex:
        """
        Current index.
        """
        return self.dates[self._counter]

    @property
    def _max_episode_steps(self) -> int:
        """
        Number of available time steps.
        """
        return len(self.dates)

    @abstractmethod
    def _get_assets_data(
        self, start: datetime, end: datetime, universe: Optional[List[str]], **kwargs
    ) -> pd.DataFrame:
        """
        Get assets prices.

        Parameters
        ----------
        universe: List[str]
            Assets universe.
        trading_period:

        Returns
        -------
        pd.DataFrame
            Prices for all the assets in the universe.
        """
        raise NotImplementedError

    def _get_observation(self) -> Dict[str, float]:
        """
        Get the current observation.
        """
        ob = {}
        ob["close"] = self.close.loc[self.index, :].dropna()
        ob["open"] = self.open.loc[self.index, :].dropna()
        ob["low"] = self.low.loc[self.index, :].dropna()
        ob["high"] = self.high.loc[self.index, :].dropna()
        ob["volume"] = self.volume.loc[self.index, :].dropna()
        ob["returns"] = self._returns.loc[self.index, :].dropna()

        return ob

    def _get_reward(self, action) -> pd.Series:
        """
        Get agent's reward.
        """
        return (self._returns.loc[self.index] * action).fillna(0.0)

    def _get_done(self) -> bool:
        """
        Returns true if there are not more steps.
        """
        return self.index == self.dates[-1]

    def _get_info(self) -> Dict:
        return {}

    def _validate_agents(self) -> None:
        """
        Checks for the availability of agents in the environment.
        """
        if len(self.agents) == 0:
            raise RuntimeError("There are no agents registered in the environment")

    @staticmethod
    def _check_agent(agent: Agent) -> None:
        if not hasattr(agent, "name"):
            raise ValueError("The agent must have an attribute 'name'.")

    def register(self, agent: Agent) -> None:
        """
        Register an agent in the environment.

        Parameters
        ----------
        agent: Agent
            Agent to register.

        Returns
        -------
        None.
        """

        self._check_agent(agent)

        if agent.name not in self.agents:
            self.agents[agent.name] = self.Record(
                columns=self.universe, index=self.dates
            )

    def unregister(self, agent: Optional[Agent] = None) -> None:
        """
        Eliminates an agent from the environment. If agent==None, all the
        agents are eliminated.

        Parameters
        ----------
        agent: Agent
            Agent to unregister.

        Returns
        -------
        None.
        """
        if agent is None:
            self.agents = {}
            return None

        self._check_agent(agent)

        if agent.name in self.agents:
            del self.agents[agent.name]

    def step(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, float], Union[float, Dict[str, float]], bool, Dict]:

        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action: Dict[str, np.ndarray]
            Portfolio vector(s).

        Returns
        -------
        observation, reward, done, info: Tuple[Dict, Union[float, Dict[str, float]], bool, Dict]
            - observation: Dict[str, float]
                Environment observation.
            - reward: float | Dict[str, float]
                Reward(s) received after the step.
            - done: bool
                Wheter the episode ended or not.
            - info: Dict
                Information about the step.
        """

        self._validate_agents()

        # For fees
        prev_index = self.index

        self._counter += 1

        observation = self._get_observation()

        done = self._get_done()

        info = self._get_info()

        if action.keys() != self.agents.keys():
            raise ValueError(
                "Invalid action interface. Action keys do not match agents keys (possible unregistered agent)."
            )

        reward = {}

        for name, _action in action.items():

            # nan_symbols = {
            #     symbol: 0.0 for symbol in self.universe if symbol not in _action.index
            # }

            nan_symbols = [
                symbol for symbol in self.universe if symbol not in _action.index
            ]

            _action = _action.append(
                pd.Series(
                    np.zeros(len(nan_symbols)),
                    index=nan_symbols,
                    name=observation["returns"].name,
                )
            )

            if not self.action_space.contains(_action):
                raise ValueError(f"Invalid action for agent {name}: {_action}")

            # self.agents[name].actions.loc[self.index] = _action
            self.agents[name].actions[self.index] = _action
            # self.agents[name].rewards.loc[self.index] = self._get_reward(_action)
            reward_ = self._get_reward(_action)

            prev_action = self.agents[name].actions[prev_index]

            commission = self.fee * (_action - prev_action).abs()

            self.agents[name].rewards[self.index] = reward_ - commission
            # reward[name] = self.agents[name].rewards.loc[self.index].sum()

            reward[name] = (reward_ - commission).sum()

            if done:
                self.agents[name].actions = pd.DataFrame.from_dict(
                    self.agents[name].actions, orient="index"
                )

                self.agents[name].rewards = pd.DataFrame.from_dict(
                    self.agents[name].rewards, orient="index"
                )

        return observation, reward, done, info

    def reset(self) -> Dict[str, float]:
        """
        Reset the environment's state and returns an initial observation.

        Returns
        -------
        observation: Dict[str, float]
            Initial observation.
        """

        self._validate_agents()
        self._counter = 0
        ob = self._get_observation()

        return ob

    def render(self) -> None:

        if self._fig is None or self._axes is None:
            self._fig, self._axes = plt.subplots(ncols=2, figsize=(12, 6))

        _pnl = pd.DataFrame(columns=self.agents.keys(), index=self.dates)

        for agent in self.agents:
            _pnl.loc[:, agent] = (
                self.agents[agent].rewards.sum(axis=1) + 1.0
            ).cumprod()

        self._axes[0].clear()

        self._axes[1].clear()

        self.close.loc[: self.index].plot(ax=self._axes[0])

        _pnl.loc[: self.index].plot(ax=self._axes[1])

        self._axes[0].set_xlim(self.dates.min(), self.dates.max())

        self._axes[0].set_title("Market Prices")

        self._axes[0].set_ylabel("Prices")

        self._axes[1].set_xlim(self._pnl.index.min(), self._pnl.index.max())

        self._axes[1].set_title("PnL")

        self._axes[1].set_ylabel("Wealth Level")

        plt.pause(0.0001)

        self._fig.canvas.draw()

    def summary(self) -> pd.DataFrame:
        """
        Generates a summary of statistics and figures.

        Returns
        -------
        table: pd.DataFrame
            Strategy report.
        """

        summary = {}

        for agent in self.agents:

            prices = self.close

            returns = self.agents[agent].rewards.sum(axis=1)

            returns.name = agent

            weights = self.agents[agent].actions

            weights.name = agent

            summary[agent] = stats(returns)

            # figure(prices, returns, weights)

        return pd.DataFrame(summary)
