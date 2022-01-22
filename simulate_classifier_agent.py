from datetime import datetime
import pickle
from simulate_agent import simulate_agent


if __name__ == "__main__":

    from trading_gym.agents.classifier import ClassifierAgent
    from trading_gym.utils.screener import Screener

    assets_data = pickle.load(open("./assets_data_training.pickle", "rb"))
    windows = [
        180,
    ]
    days_ahead = 1
    target_type_and_q = [
        ("binary", None),
        ("binary", 3),
        ("binary", 4),
        ("binary", 5),
        ("multiclass", 3),
        ("multiclass", 4),
        ("multiclass", 5),
    ]
    q = None
    reduce_dimensionality_list = [True, False]
    apply_wavelet_decomp_list = [True, False]
    q = [2, 3, 4, 5]
    n_assets_list = [5, 10]
    # screeners = [
    #     Screener("returns", 10, 15),
    #     Screener("returns", 5, 15),
    #     Screener("returns", 10, 5),
    #     Screener("returns", 5, 5),
    # ]
    retrain_each_n_obs_list = [15, 30]
    params_grid = []
    for w in windows:
        for target_type, q in target_type_and_q:
            for reduce_dimensionality in reduce_dimensionality_list:
                for apply_wavelet_decomp in apply_wavelet_decomp_list:
                    for retrain_each_n_obs in retrain_each_n_obs_list:
                        for n_assets in n_assets_list:
                            params_grid.append(
                                {
                                    "window": w,
                                    "screener": None,
                                    "target_type": target_type,
                                    "q": q,
                                    "reduce_dimensionality": reduce_dimensionality,
                                    "apply_wavelet_decomp": apply_wavelet_decomp,
                                    "retrain_each_n_obs": retrain_each_n_obs,
                                    "n_assets": n_assets,
                                }
                            )
    start = datetime(2019, 5, 1)
    end = datetime(2021, 9, 30)
    start_eval_date = datetime(2020, 1, 1)
    simulate_agent(
        assets_data=assets_data,
        start=start,
        end=end,
        agent_class=ClassifierAgent,
        agent_params_grid=params_grid,
        file_suffix="no_screener",
    )
