from datetime import datetime
import pickle
from simulate_agent import simulate_agent, parallel_simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.tipp import TippAgent
    from trading_gym.agents.hrp_riskfoliolib import HRPAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))

    # TRAINING
    # start = datetime(2018, 5, 1)
    # end = datetime(2021, 1, 1)
    # params_grid = []
    # start_eval_date = datetime(2021, 1, 1)
    # params_grid = []
    # multipliers = list(range(3, 10))
    # floor_pcts = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # w_risk_min_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # w_risk_max_list = [1.0, 0.95, 0.90, 0.85]
    # for multiplier in multipliers:
    #     for floor_pct in floor_pcts:
    #         for w_risk_min in w_risk_min_list:
    #             for w_risk_max in w_risk_max_list:
    #                 params_grid.append(
    #                     {
    #                         "multiplier": multiplier,
    #                         "floor_pct": floor_pct,
    #                         "w_risk_min": w_risk_min,
    #                         "w_risk_max": w_risk_max,
    #                         "window": 180,
    #                     }
    #                 )
    # TESTING WEEKLY REB
    start = datetime(2020, 5, 1)
    end = datetime(2022, 1, 17)
    params_grid = [
        {"multiplier": 9, "floor_pct": 0.75, "window": 180, "w_risk_min": 0.05}
    ]
    start_eval_date = datetime(2021, 1, 1)
    # start = datetime(2020, 5, 1)
    # end = datetime(2022, 1, 12)
    # params_grid = [
    #     {
    #         "multiplier": 8,
    #         "floor_pct": 0.75,
    #         "window": 180,
    #         "w_risk_min": 0.40,
    #         "w_risk_max": 0.90,
    #     }
    # ]
    # start_eval_date = datetime(2021, 1, 1)

    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        agent_class=TippAgent,
        agent_params_grid=params_grid,
        tipp_agent=HRPAgent,
        fee=0.01,
        tipp_agent_params={
            "window": 180,
            "screener": [Screener("volume", 200, 15), Screener("returns", 10, 15)],
            "model": "HRP",
            "rebalance_each_n_obs": 7,
            "codependence": "pearson",
            "covariance": "hist",
            "objective": "Sharpe",
            "risk_measure": "MV",
            "leaf_order": True,
            "w_min": 0.05,
            "w_max": 0.35,
        },
        file_suffix="testing_today",
    )
