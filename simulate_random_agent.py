from datetime import datetime
import pickle
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.random import RandomAgent

    random_universes = pickle.load(open("./random_universes.pickle", "rb"))
    params_grid = [{"window": 50}]
    start = datetime(2019, 5, 1)
    end = datetime(2021, 9, 30)
    start_eval_date = datetime(2020, 1, 1)
    simulate_agent(
        random_universes=random_universes[:10],
        start=start,
        end=end,
        agent_class=RandomAgent,
        agent_params_grid=params_grid,
        start_eval_date=start_eval_date,
    )
