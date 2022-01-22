from datetime import datetime
import pickle
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.genetic import GeneticAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))
    # windows = [180, 60, 30, 15]
    # screeners = [
    #     Screener("mix", 10, 15),
    #     Screener("mix", 5, 15),
    #     Screener("mix", 10, 5),
    #     Screener("mix", 5, 5),
    # ]
    # generations = 500
    # pop_size = 500
    # params_grid = []
    # for w in windows:
    #     for screener in screeners:
    #         params_grid.append(
    #             {
    #                 "window": w,
    #                 "generations": generations,
    #                 "pop_size": pop_size,
    #                 "screener": screener,
    #                 "rebalance_each_n_obs": 7,
    #             }
    #         )

    # start = datetime(2019, 5, 1)
    # end = datetime(2021, 9, 30)

    # TESTING
    start = datetime(2020, 5, 1)
    end = datetime(2022, 1, 17)
    params_grid = [
        {
            "window": 180,
            "generations": 500,
            "pop_size": 500,
            "screener": [Screener("volume", 200, 15), Screener("returns", 10, 15)],
        }
    ]
    start_eval_date = datetime(2021, 1, 1)

    # TESTING WEEKLY REB
    # start = datetime(2020, 5, 1)
    # end = datetime(2022, 1, 12)
    # params_grid = [
    #     {
    #         "window": 15,
    #         "generations": 500,
    #         "pop_size": 500,
    #         "screener": Screener("mix", 5, 5),
    #         "rebalance_each_n_obs": 7,
    #     }
    # ]
    # start_eval_date = datetime(2020, 1, 1)

    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        fee=0.01,
        agent_class=GeneticAgent,
        agent_params_grid=params_grid,
        file_suffix="testing_today",
    )
