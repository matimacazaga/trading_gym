from datetime import datetime
import pickle
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.genetic import GeneticAgent
    from trading_gym.agents.tipp import TippAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))

    # TESTING
    start = datetime(2020, 5, 1)
    end = datetime(2022, 1, 12)
    params_grid = [
        {
            "multiplier": 5,
            "floor_pct": 0.9,
            "window": 180,
        },
    ]
    start_eval_date = datetime(2020, 1, 1)

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
        agent_class=TippAgent,
        agent_params_grid=params_grid,
        tipp_agent=GeneticAgent,
        tipp_agent_params={
            "window": 180,
            "generations": 500,
            "pop_size": 500,
            "screener": Screener("returns", 10, 15),
        },
        file_suffix="testing_oot",
    )
