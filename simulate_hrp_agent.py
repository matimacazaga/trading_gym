from datetime import datetime
import pickle
from simulate_agent import parallel_simulate_agent, simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.hrp import HRPAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))
    # windows = [180, 90, 60, 30]
    # screeners = []
    # for i in range(5, 20):
    #     for j in range(5, 30):
    #         screeners.append(Screener("returns", i, j))

    # params_grid = []
    # for w in windows:
    #     for screener in screeners:
    #         params_grid.append(
    #             {"window": w, "screener": screener, "rebalance_each_n_obs": 7}
    #         )
    # start = datetime(2019, 1, 1)
    # end = datetime(2021, 1, 1)
    # start_eval_date = datetime(2021, 1, 1)

    # TESTING
    # params_grid = [{"window": 180, "screener": Screener("returns", 10, 15)}]
    # start = datetime(2020, 5, 1)
    # end = datetime(2022, 1, 12)

    # TESTING WEEKLY REB
    params_grid = [
        {
            "window": 180,
            "screener": Screener("returns", 10, 15),
            "rebalance_each_n_obs": 7,
        },
    ]

    start = datetime(2020, 5, 1)
    end = datetime(2022, 1, 12)

    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        fee=0.005,
        agent_class=HRPAgent,
        cash=False,
        agent_params_grid=params_grid,
        file_suffix="testing_oot_0005",
    )
