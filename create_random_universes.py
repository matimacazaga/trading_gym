from typing import List
from trading_gym.envs.data_loader import CryptoData
import numpy as np
from datetime import datetime
from tqdm import tqdm


def get_random_universes(
    n_symbols: int, n_universes: int, start: datetime, end: datetime
) -> List[List[str]]:
    """
    Get random universes for evaluating the agents.

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
        if (
            symbol.endswith("USDT")
            or symbol.endswith("BUSD")
            or symbol.endswith("USDC")
        )
        and not symbol.startswith("USDT")
        and not symbol.startswith("USDS")
        and not symbol.startswith("BUSD")
        and not symbol.startswith("USDC")
        and not symbol.startswith("TUSD")
        and not symbol.startswith("DAI")
        and not symbol.startswith("PAX")
        and not symbol.startswith("EUR")
        and not symbol.startswith("MDX")
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

    symbols = list(filter(lambda s: s in assets_data["close"].columns, symbols))

    random_universes = []

    print("Creating universes")
    for _ in tqdm(range(n_universes)):

        universe = list(np.random.choice(symbols, size=n_symbols, replace=False))

        universe_data = {}
        for k in assets_data:
            universe_data[k] = assets_data[k].loc[:, universe].dropna(how="all", axis=1)

        random_universes.append(universe_data)

    print("Done")

    return random_universes


if __name__ == "__main__":

    import pickle

    start = datetime(2019, 5, 1)
    end = datetime(2021, 9, 30)

    random_universes = get_random_universes(50, 100, start, end)

    pickle.dump(random_universes, open("./random_universes.pickle", "wb"))
