from datetime import datetime
from typing import List, Optional
from .data_loader import CryptoData
from .base import BaseEnv
import pandas as pd


class TradingEnv(BaseEnv):
    def _get_assets_data(
        self, start: datetime, end: datetime, universe: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return CryptoData().get_data(start, end, universe)


if __name__ == "__main__":

    trading_env = TradingEnv(
        universe=["BTCUSDT", "ETHUSDT"],
        start=datetime(2021, 12, 1),
        end=datetime(2021, 12, 21),
    )

    print(trading_env.reset())
