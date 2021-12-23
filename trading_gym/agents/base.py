import numpy as np
from abc import abstractmethod
from abc import ABCMeta

# from ..utils.gym import run


class Agent(metaclass=ABCMeta):

    _id = "base"

    def __init__(self, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return self._id

    def begin_episode(sellf, observation):
        pass

    @abstractmethod
    def act(self, observation):
        return

    def observe(self, observation, action, reward, done, next_reward):
        pass

    def end_episode(self):
        pass

    # def fit(self, env, num_episodes=1, verbose=False):
    #     return run(env, self, num_episodes, True, verbose)
