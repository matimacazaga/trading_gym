from typing import List
from trading_gym.envs.data_loader import CryptoData
import numpy as np
from datetime import datetime
from tqdm import tqdm


def get_assets_data(start: datetime, end: datetime) -> List[List[str]]:
    """
    Get assets data.

    Parameters
    ----------
    n_symbols: int
        Number of symbols in each universe.
    n_universes: int
        Number of universes to generate.
    start: datetime
        Start date.
    end: datetime
        End date.
    Returns
    -------
    random_universes: List[List[str]]
        List of random universes.
    """
    cd = CryptoData()

    supported_symbols = cd.supported_symbols

    symbols_filtered = [
        symbol
        for symbol in supported_symbols
        if (symbol.endswith("USDT") or symbol.endswith("BUSD"))
        and not symbol.startswith("USDT")
        and not symbol.startswith("SUSD")
        and not symbol.startswith("USDS")
        and not symbol.startswith("BUSD")
        and not symbol.startswith("USDC")
        and not symbol.startswith("TUSD")
        and not symbol.startswith("DAI")
        and not symbol.startswith("PAX")
        and not symbol.startswith("EUR")
        and not symbol.startswith("MDX")
        and not symbol.startswith("AUD")
        and not symbol.startswith("GBP")
        and symbol
        not in [
            "UNIDOWNUSDT",
            "LTCDOWNUSDT",
            "SUSHIDOWNUSDT",
            "ETHDOWNUSDT",
            "BNBDOWNUSDT",
            "XLMDOWNUSDT",
            "SXPDOWNUSDT",
            "TRXDOWNUSDT",
            "YFIDOWNUSDT",
            "XTZDOWNUSDT",
            "AAVEDOWNUSDT",
            "DOTDOWNUSDT",
            "LINKDOWNUSDT",
            "ADADOWNUSDT",
            "XRPDOWNUSDT",
            "COCOSUSDT",
        ]
    ]

    symbols = {}

    for symbol in symbols_filtered:

        if "USDT" in symbol:
            quote = "USDT"
        elif "BUSD" in symbol:
            quote = "BUSD"
        elif "USDC" in symbol:
            quote = "USDC"
        else:
            continue

        s = symbol.replace(quote, "")
        if s in symbols:
            continue
        else:
            symbols[s] = symbol

    symbols = list(symbols.values())

    print("Downloading data")

    assets_data = cd.get_data(start=start, end=end, universe=symbols)

    return assets_data


if __name__ == "__main__":

    import pickle
    from datetime import datetime
    import pandas as pd

    # TRAINING V1
    # start = datetime(2019, 5, 1)
    # end = datetime(2021, 9, 30)

    # TRAINING V2
    # start = datetime(2018, 5, 1)
    # end = datetime(2021, 1, 1)
    # TESTING V1
    start = datetime(2020, 5, 1)
    end = datetime(2022, 1, 21)

    assets_data = get_assets_data(start, end)

    pickle.dump(assets_data, open("./assets_data_testing.pickle", "wb"))

    # UPDATE EXISTING TESTING DATA
    # start = datetime(2022, 1, 12)
    # end = datetime(2022, 1, 17)

    # assets_data = pickle.load(open("./assets_data_testing.pickle", "rb"))

    # tmp = get_assets_data(start, end)

    # assets_data_updated = {}

    # for k in assets_data:
    #     assets_data_updated[k] = pd.concat([assets_data[k], tmp[k]])

    # pickle.dump(assets_data_updated, open("./assets_data_testing.pickle", "wb"))
