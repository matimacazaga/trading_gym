from datetime import datetime
import pickle
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.dlpopt import DeepLPortfolioAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))
    # windows = [180, 60, 30, 15]
    # epochs = 200
    # screeners = [
    #     Screener("mix", 10, 15),
    #     Screener("mix", 5, 15),
    #     Screener("mix", 10, 5),
    #     Screener("mix", 5, 5),
    # ]
    # params_grid = []
    # for w in windows:
    #     for screener in screeners:
    #         params_grid.append(
    #             {
    #                 "window": w,
    #                 "hidden_units": 50,
    #                 "epochs": epochs,
    #                 "screener": screener,
    #                 "rebalance_each_n_obs": 7,
    #             }
    #         )
    # start = datetime(2019, 5, 1)
    # end = datetime(2021, 9, 30)
    # start_eval_date = datetime(2020, 1, 1)

    # TESTING
    params_grid = [
        {
            "window": 180,
            "screener": Screener("returns", 10, 15),
            "hidden_units": 50,
            "epochs": 200,
        }
    ]
    start = datetime(2020, 5, 1)
    end = datetime(2022, 1, 12)

    # TESTING WEEKLY REB
    # params_grid = [
    #     {
    #         "window": 15,
    #         "screener": Screener("mix", 5, 5),
    #         "hidden_units": 50,
    #         "epochs": 200,
    #         "rebalance_each_n_obs": 7,
    #     }
    # ]
    # start = datetime(2020, 5, 1)
    # end = datetime(2022, 1, 12)

    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        agent_class=DeepLPortfolioAgent,
        agent_params_grid=params_grid,
        file_suffix="testing_oot",
    )
