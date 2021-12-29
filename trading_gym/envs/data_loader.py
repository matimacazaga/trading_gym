from typing import List, Optional
import pandas as pd
from datetime import datetime
from .binance_wrapper import BinanceWrapper
import warnings
from joblib import Parallel, delayed


class CryptoData:
    """
    Wrapper for cryptocurrencies data.
    """

    def __init__(self):

        self.client = BinanceWrapper()

        self._cols = ["close", "open", "high", "low", "volume"]

        self._source = "Binance"

        self._supported_symbols = self.client.get_crypto_list()

    @property
    def source(self) -> str:
        return self._source

    @property
    def supported_symbols(self) -> List[str]:
        return self._supported_symbols

    def _get_symbol_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ):

        days = (end - start).days

        if days > 1000:
            limits = days // 999
            limits = [999 for _ in range(limits)]
            if modulo := days % 999:
                limits += [modulo]
            dfs = []
            for l, lim in enumerate(limits):
                dfs.append(
                    self.client.get_klines(
                        symbol,
                        "1d",
                        limit=lim,
                        start_time=start
                        if l == 0
                        else dfs[-1].iloc[-1]["close_time"].to_pydatetime(),
                    )
                )

            df = pd.concat(dfs)

            df.loc[:, "close_time"] = df.loc[:, "close_time"].apply(
                lambda x: datetime(x.year, x.month, x.day)
            )

            df.drop_duplicates(subset="close_time", keep="first", inplace=True)

        else:

            df = self.client.get_klines(
                symbol,
                "1d",
                limit=days,
                start_time=start,
                end_time=end,
            )

            df.loc[:, "close_time"] = df.loc[:, "close_time"].apply(
                lambda x: datetime(x.year, x.month, x.day)
            )

            df.drop_duplicates(subset="close_time", keep="last", inplace=True)

        df.drop("volume", axis=1, inplace=True)

        df.rename(
            {"close_time": "date", "quote_asset_vol": "volume"},
            axis=1,
            inplace=True,
        )

        df.drop(
            [col for col in df.columns if col not in self._cols + ["date"]],
            axis=1,
            inplace=True,
        )

        df.set_index("date", inplace=True)

        return df

    def _check_symbol(self, symbol: str) -> bool:
        if symbol in self.supported_symbols:

            return True
        else:
            warnings.warn(
                f"The symbol {symbol} is discarded because it is not supported.",
                UserWarning,
            )

            return False

    def get_data(
        self, start: datetime, end: datetime, universe: Optional[List[str]] = None
    ):

        close = {}
        open = {}
        high = {}
        low = {}
        volume = {}

        if universe:

            universe = list(filter(lambda s: self._check_symbol(s), universe))

        else:

            universe = self.supported_symbols

        dfs = Parallel(n_jobs=min(40, len(universe)), backend="threading")(
            delayed(self._get_symbol_data)(symbol, start, end) for symbol in universe
        )
        for symbol, temp_df in zip(universe, dfs):
            tmp_df = self._get_symbol_data(symbol, start, end)

            if tmp_df is not None and len(tmp_df) > 0:
                close[symbol] = tmp_df.loc[:, "close"]
                open[symbol] = tmp_df.loc[:, "open"]
                high[symbol] = tmp_df.loc[:, "high"]
                low[symbol] = tmp_df.loc[:, "low"]
                volume[symbol] = tmp_df.loc[:, "volume"]

        data = {
            "close": pd.DataFrame(close).sort_index(ascending=True),
            "open": pd.DataFrame(open).sort_index(ascending=True),
            "high": pd.DataFrame(high).sort_index(ascending=True),
            "low": pd.DataFrame(low).sort_index(ascending=True),
            "volume": pd.DataFrame(volume).sort_index(ascending=True),
        }

        return data


if __name__ == "__main__":

    crypto_data = CryptoData()

    print(len(crypto_data.supported_symbols))

    data = crypto_data.get_data(
        datetime(2021, 12, 1),
        datetime(2021, 12, 21),
        universe=["BTCUSDT", "ETHUSDT", "ARGYUSDT"],
    )

    print(data.keys())

    for df in data.values():
        print(df.head())
