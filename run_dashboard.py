if __name__ == "__main__":

    from trading_gym.utils.dashboard import run_app
    from streamlit import cli as stcli
    import streamlit as st
    import sys

    FILE_PATH = (
        # "./simulation_results/hrp_combined/results_hrp_combined_params_0_testing.pickle"
    )

    FILE_PATH = "./simulation_results/hrp_riskfolio/results_hrp_riskfolio_params_0_testing_today.pickle"

    # FILE_PATH = "./simulation_results/tipp_hrp_riskfolio/results_tipp_hrp_riskfolio_params_0_testing_today.pickle"

    if st._is_running_with_streamlit:
        run_app(FILE_PATH)
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
