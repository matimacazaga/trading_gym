from datetime import datetime
import pickle
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.equal_weight import EqualWeightAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))
    # screeners = [
    #     Screener("mix", 10, 15),
    #     Screener("mix", 5, 15),
    #     Screener("mix", 10, 5),
    #     Screener("mix", 5, 5),
    # ]
    # params_grid = []
    # for screener in screeners:
    #     params_grid.append(
    #         {"window": 50, "screener": screener, "rebalance_each_n_obs": 7}
    #     )
    # start = datetime(2021, 5, 1)
    # end = datetime(2021, 9, 30)
    # start_eval_date = datetime(2020, 1, 1)

    # TESTING
    start = datetime(2020, 5, 1)
    end = datetime(2022, 1, 12)
    params_grid = [{"window": 180, "screener": Screener("returns", 10, 15)}]

    # TESTING WEEKLY REB
    # start = datetime(2020, 5, 1)
    # end = datetime(2022, 1, 12)
    # params_grid = [{"window": 50, "screener": Screener("mix", 10, 5)}]

    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        fee=0.01,
        cash=False,
        agent_class=EqualWeightAgent,
        agent_params_grid=params_grid,
        file_suffix="testing_oot_001",
    )
