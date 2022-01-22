from datetime import datetime
import pickle
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.rnn import RnnLSTMAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_training.pickle", "rb"))
    windows = [180, 60]
    past_n_obs_list = [10, 5]
    retrain_each_n_obs_list = [30, 15]
    epochs = 50
    screeners = [
        Screener("returns", 10, 15),
        Screener("returns", 5, 15),
        Screener("returns", 10, 5),
        Screener("returns", 5, 5),
    ]
    params_grid = []
    for w in windows:
        for past_n_obs in past_n_obs_list:
            for retrain_each_n_obs in retrain_each_n_obs_list:
                for screener in screeners:
                    params_grid.append(
                        {
                            "window": w,
                            "epochs": epochs,
                            "screener": screener,
                            "past_n_obs": past_n_obs,
                            "retrain_each_n_obs": retrain_each_n_obs,
                        }
                    )
    start = datetime(2019, 5, 1)
    end = datetime(2021, 9, 30)
    start_eval_date = datetime(2020, 1, 1)
    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        agent_class=RnnLSTMAgent,
        agent_params_grid=params_grid,
        file_suffix="returns",
    )
