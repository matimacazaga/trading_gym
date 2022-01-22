from typing import List, Optional
import pandas as pd


class Screener:
    def __init__(self, filter_by: str = "volume", n_assets: int = 10, n_obs: int = 15):

        self.filter_by = filter_by
        self.n_assets = n_assets
        self.n_obs = n_obs

    def _filter_by_volume(self, volume: pd.DataFrame) -> List[str]:

        return (
            volume.iloc[-self.n_obs :]
            .mean(axis=0)
            .sort_values(ascending=False)
            .index[: self.n_assets]
            .tolist()
        )

    def _filter_by_volatility(self, returns: pd.DataFrame) -> List[str]:

        return (
            returns.iloc[-self.n_obs :]
            .std(axis=0, ddof=1)
            .sort_values(ascending=True)
            .index[: self.n_assets]
            .tolist()
        )

    def _filter_by_return(self, returns: pd.DataFrame) -> List[str]:

        return (
            returns.iloc[-self.n_obs :]
            .mean(axis=0)
            .sort_values(ascending=False)
            .index[: self.n_assets]
            .tolist()
        )

    def filter(
        self,
        returns: Optional[pd.DataFrame] = None,
        volume: Optional[pd.DataFrame] = None,
    ):

        if returns is None and volume is None:
            raise ValueError("At least 'returns' or 'volume' must be provided.")

        assets_list = []

        if self.filter_by == "volatility":

            assets_list.extend(self._filter_by_volatility(returns))

        elif self.filter_by == "volume":

            assets_list.extend(self._filter_by_volume(volume))

        elif self.filter_by == "returns":

            assets_list.extend(self._filter_by_return(returns))

        elif self.filter_by == "mix_volume_returns":
            assets_vol = self._filter_by_volume(volume)
            assets_ret = self._filter_by_return(returns)
            n_assets = min(self.n_assets, len(assets_ret))
            for i in range(n_assets):
                if assets_vol[i] not in assets_list:
                    assets_list.append(assets_vol[i])
                if assets_ret[i] not in assets_list:
                    assets_list.append(assets_ret[i])
                if len(assets_list) == self.n_assets:
                    break

        elif self.filter_by == "mix_volatility_returns":
            assets_volatility = self._filter_by_volatility(returns)
            assets_ret = self._filter_by_return(returns)
            n_assets = min(self.n_assets, len(assets_ret))
            for i in range(n_assets):
                if assets_volatility[i] not in assets_list:
                    assets_list.append(assets_volatility[i])
                if assets_ret[i] not in assets_list:
                    assets_list.append(assets_ret[i])
                if len(assets_list) == self.n_assets:
                    break
        else:
            raise ValueError(f"Filter by {self.filter_by} not supported")

        return assets_list
