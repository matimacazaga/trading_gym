from datetime import datetime
import pickle
from simulate_agent import parallel_simulate_agent, simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.genetic_hrp_agent import GeneticHRPAgent

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))

    params_grid = [
        {
            "genome_spec": {
                "n_assets_volatility": (int, 30, 100),
                "n_obs_volatility": (int, 5, 15),
                "n_assets_returns": (int, 5, 15),
                "n_obs_returns": (int, 5, 15),
            },
            "cash": False,
            "generations": 50,
            "pop_size": 48,
            "n_obs_volume": 15,
            "n_assets_volume": 200,
        }
    ]

    start = datetime(2018, 5, 1)
    end = datetime(2021, 1, 1)
    start_eval_date = datetime(2019, 1, 1)

    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        fee=0.01,
        agent_class=GeneticHRPAgent,
        cash=False,
        agent_params_grid=params_grid,
        file_suffix="testing_cum_return",
    )
