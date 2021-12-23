from typing import List, Tuple
import numpy as np
from gym import Env
from ..agents.base import Agent
import os


def run(
    env: Env, agent: Agent, num_episodes: int, record: bool = True, log: bool = False
):

    if hasattr(env, "unregister"):
        env.unregister(agent=None)

    if hasattr(env, "register"):

        if not hasattr(agent, "name"):
            agent.name = "_default"

        env.register(agent)

    rewards = []

    actions = []

    _best_reward = -np.inf

    def _run() -> Tuple[List[float], List[np.ndarray]]:

        _rewards = []

        _actions = []

        ob = env.reset()

        reward = 0.0

        done = False

        info = {}

        j = 0

        agent.begin_episode(ob)

        while (not done) and (j < env._max_episode_steps):

            action = agent.act(ob)

            if hasattr(env, "register"):

                ob_, reward, done, info = env.step({agent.name: action})

                reward = reward[agent.name]

            else:

                ob_, reward, done, info = env.step(action)

            _rewards.append(reward)

            _actions.append(action)

            agent.observe(ob, action, reward, done, ob_)

            ob = ob_

            j += 1

        agent.end_episode()

        return _rewards, _actions

    for e in range(num_episodes):

        _rewards, _actions = _run()

        if record:

            rewards.append(_rewards)

            actions.append(_actions)

        if log:
            print(f"episode: {e:4d}, cumulative reward: {sum(_rewards):.5f}")

        if sum(_rewards) > _best_reward:

            try:

                os.remove(f"tmp/models/{agent.name}/{_best_reward}.h5")

            except:
                pass

            _best_reward = sum(_rewards)

            if hasattr(agent, "save"):

                if not os.path.exists(f"tmp/models/{agent.name}"):
                    os.makedirs(f"tmp/models/{agent.name}")

                agent.save(f"tmp/models/{agent.name}/{_best_reward}.h5")
