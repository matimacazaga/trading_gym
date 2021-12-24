from datetime import datetime
from ..utils.summary import stats
from ..agents.rnn import RnnLSTMAgent
from ..envs.trading import TradingEnv
import pickle
from tqdm import tqdm


def sim_rnn_lstm_agent(universe, window, past_n_obs, retrain_each_n_obs=5):

    env = TradingEnv(
        universe=universe,
        cash=False,
        start=datetime(2020, 1, 1),
        end=datetime(2021, 12, 21),
    )

    agent = RnnLSTMAgent(
        action_space=env.action_space,
        window=window,
        past_n_obs=past_n_obs,
        retrain_each_n_obs=retrain_each_n_obs,
    )

    env.register(agent)

    ob = env.reset()

    ob = ob

    reward = 0

    done = False

    for _ in tqdm(range(env._max_episode_steps)):
        if done:
            break
        agent.observe(ob, None, reward, done, None)
        action = agent.act(ob)
        ob, reward, done, _ = env.step({agent.name: action})
        ob = ob

    pickle.dump(
        env.agents[agent.name],
        open("./trading_gym/testing/tests_results/rnn_lstm_agent_test.pickle", "wb"),
    )

    pickle.dump(
        stats(env.agents[agent.name].rewards.iloc[window:].sum(axis=1)),
        open("./trading_gym/testing/tests_results/rnn_lstm_agent_stats.pickle", "wb"),
    )


if __name__ == "__main__":

    UNIVERSE = ["BTCUSDT", "ETHUSDT"]
    WINDOW = 180
    PAST_N_OBS = 10
    RETRAIN_EACH_N_OBS = 30
    sim_rnn_lstm_agent(UNIVERSE, WINDOW, PAST_N_OBS, RETRAIN_EACH_N_OBS)
