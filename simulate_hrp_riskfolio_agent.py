from datetime import datetime
import pickle
from simulate_agent import parallel_simulate_agent, simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.hrp_riskfoliolib import HRPAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))
    # windows = [260, 180, 60, 30]
    # screeners = [
    #     Screener("returns", 10, 15),
    #     Screener("returns", 5, 15),
    #     Screener("returns", 10, 5),
    #     Screener("returns", 5, 5),
    # ]

    # model = "HRP"
    # codependences = [
    #     "pearson",
    #     "spearman",
    #     "abs_pearson",
    #     "abs_spearman",
    #     "distance",
    #     "mutual_info",
    #     "tail",
    # ]
    # covariances = ["hist", "ewma1", "ewma2", "ledoit", "oas", "shrunk"]
    # objectives = ["MinRisk", "Sharpe"]
    # risk_measures = [
    #     "MV",
    #     "MSV",
    #     "FLPM",
    #     "SLPM",
    #     "VaR",
    #     "CVaR",
    #     "WR",
    #     "MDD",
    # ]
    # leaf_orders = [True, False]
    # params_grid = []
    # for w in windows:
    #     for screener in screeners:
    #         for codependence in codependences:
    #             for covariance in covariances:
    #                 for objective in objectives:
    #                     for risk_measure in risk_measures:
    #                         for leaf_order in leaf_orders:
    # params_grid.append(
    #     {
    #         "window": w,
    #         "screener": screener,
    #         "model": model,
    #         "rebalance_each_n_obs": 7,
    #         "codependence": codependence,
    #         "covariance": covariance,
    #         "objective": objective,
    #         "risk_measure": risk_measure,
    #         "leaf_order": leaf_order,
    #     }
    # )

    params_grid = [
        {
            "window": 180,
            "screener": [
                Screener("volume", 200, 15),
                Screener("volatility", 50, 15),
                Screener("returns", 10, 15),
            ],
            "model": "HRP",
            "rebalance_each_n_obs": 7,
            "codependence": "pearson",
            "covariance": "hist",
            "objective": "Sharpe",
            "risk_measure": "MV",
            "leaf_order": True,
            "w_min": 0.05,
            "w_max": 0.35,
        }
    ]
    start = datetime(2019, 5, 1)
    end = datetime(2022, 1, 17)
    start_eval_date = datetime(2021, 1, 1)

    # n_obs_volatility = [5, 15, 30]
    # n_obs_returns = [5, 15, 30]
    # n_assets_volatility = [30, 60, 90]
    # n_assets_returns = [5, 10, 15, 20]
    # windows = [180, 60, 30]
    # params_grid = []
    # for n_obs_vol in n_obs_volatility:
    #     for n_obs_ret in n_obs_returns:
    #         for n_assets_vol in n_assets_volatility:
    #             for n_assets_ret in n_assets_returns:
    #                 for window in windows:
    #                     params_grid.append(
    #                         {
    #                             "window": window,
    #                             "screener": [
    #                                 Screener("volume", 200, 15),
    #                                 Screener("volatility", n_assets_vol, n_obs_vol),
    #                                 Screener("returns", n_assets_ret, n_obs_ret),
    #                             ],
    #                             "model": "HRP",
    #                             "rebalance_each_n_obs": 7,
    #                             "codependence": "pearson",
    #                             "covariance": "hist",
    #                             "objective": "Sharpe",
    #                             "risk_measure": "MDD",
    #                             "leaf_order": True,
    #                             "w_min": 0.05,
    #                             "w_max": 0.35,
    #                         }
    #                     )
    # start = datetime(2018, 5, 1)
    # end = datetime(2021, 1, 1)
    # start_eval_date = datetime(2019, 1, 1)

    # TESTING
    # params_grid = [{"window": 180, "screener": Screener("returns", 10, 15)}]
    # start = datetime(2020, 5, 1)
    # end = datetime(2022, 1, 12)

    # TESTING WEEKLY REB
    # params_grid = [
    #     {
    #         "window": 180,
    #         "screener": Screener("returns", 10, 15),
    #         "codependence": "pearson",
    #         "objective": "Sharpe",
    #         "rebalance_each_n_obs": 7,
    #     },
    # ]
    # start = datetime(2020, 5, 1)
    # end = datetime(2022, 1, 12)

    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        fee=0.01,
        agent_class=HRPAgent,
        cash=True,
        agent_params_grid=params_grid,
        file_suffix="testing_today",
    )
