from ..utils.screener import Screener
from .base import Agent
from ..envs.spaces import PortfolioVector
from collections import deque
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import talib
import pywt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class ClassifierAgent(Agent):
    _id = "classifier"

    def __init__(
        self,
        action_space: PortfolioVector,
        window: int,
        days_ahead: int = 1,
        target_type: str = "binary",
        q: Optional[int] = None,
        reduce_dimensionality: bool = True,
        apply_wavelet_decomp: bool = False,
        screener: Optional[Screener] = None,
        n_assets: Optional[int] = None,
        rebalance_each_n_obs: int = 1,
        retrain_each_n_obs: int = 15,
        *args,
        **kwargs
    ):

        self.action_space = action_space
        self.observation_size = self.action_space.shape[0]
        self.memory_close = deque(maxlen=window)
        self.memory_high = deque(maxlen=window)
        self.memory_low = deque(maxlen=window)
        self.memory_volume = deque(maxlen=window)
        self.memory_returns = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.days_ahead = days_ahead
        self.target_type = target_type
        self.q = q
        self.n_assets = n_assets
        self.scaler = StandardScaler()
        self.reduce_dimensionality = reduce_dimensionality
        if reduce_dimensionality:
            self.pca = PCA(n_components=0.95, random_state=42)
        self.apply_wavelet_decomp = apply_wavelet_decomp
        self.screener = screener
        self.rebalance_each_n_obs = rebalance_each_n_obs
        self.rebalance_counter = 0
        self.retrain_each_n_obs = retrain_each_n_obs
        self.retrain_counter = 0

    def observe(self, observation: Dict[str, pd.Series], *args, **kwargs):

        self.memory_close.append(observation["close"])
        self.memory_returns.append(observation["returns"])
        self.memory_low.append(observation["low"])
        self.memory_high.append(observation["high"])
        self.memory_volume.append(observation["volume"])

    @staticmethod
    def _get_macd(close: Union[np.ndarray, pd.Series]) -> np.ndarray:

        macd, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26)

        return macd

    @staticmethod
    def _get_ht_sine(close: np.ndarray) -> np.ndarray:

        sine, _ = talib.HT_SINE(close)

        return sine

    def create_target(
        self,
        close: pd.DataFrame,
        n: int = 1,
        target_type: str = "binary",
        q: Optional[int] = None,
    ):
        target = (close.shift(-n) / close) - 1.0
        target = (
            target.stack()
            .reset_index()
            .rename({"level_1": "symbol", 0: "target"}, axis=1)
        )
        if target_type == "binary":
            if q is None:
                target.loc[:, "target"] = target.loc[:, "target"].apply(
                    lambda x: 1 if x > 0 else 0
                )
            else:
                target.loc[:, "target"] = (
                    target.groupby("date")
                    .apply(lambda x: pd.qcut(x["target"], q=q, labels=False))
                    .values
                )
                target.loc[:, "target"] = target.loc[:, "target"].apply(
                    lambda x: 1 if x == q - 1 else 0
                )
        else:
            if q is None:
                raise ValueError("q must be provided if target_type='multi'")
            target.loc[:, "target"] = (
                target.groupby("date")
                .apply(lambda x: pd.qcut(x["target"], q=q, labels=False))
                .values
            )

        target.rename({"level_0": "date"}, axis=1, inplace=True)

        return target

    def preprocess_data(
        self,
        close: pd.DataFrame,
        low: pd.DataFrame,
        high: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:

        momentum = close.apply(lambda x: talib.MOM(x, timeperiod=14))
        macd = close.apply(self._get_macd)
        mfi = close.apply(
            lambda x: talib.MFI(
                high.loc[x.index, x.name],
                low.loc[x.index, x.name],
                x,
                volume.loc[x.index, x.name],
            )
        )
        natr = close.apply(
            lambda x: talib.NATR(
                high.loc[x.index, x.name], low.loc[x.index, x.name], x
            ),
        )

        htdcp = close.apply(talib.HT_DCPHASE)
        hts = close.apply(self._get_ht_sine)
        httmm = close.apply(talib.HT_TRENDMODE)
        obv = close.apply(lambda x: talib.OBV(x, volume.loc[x.index, x.name]))
        co = close.apply(
            lambda x: talib.ADOSC(
                high.loc[x.index, x.name],
                low.loc[x.index, x.name],
                x,
                volume.loc[x.index, x.name],
            )
        )

        features_dfs = [
            ("momentum", momentum),
            ("macd", macd),
            ("mfi", mfi),
            ("natr", natr),
            ("htdcp", htdcp),
            ("hts", hts),
            ("httmm", httmm),
            ("on_balance_vol", obv),
        ]

        df = (
            co.stack()
            .reset_index()
            .rename(
                {"level_0": "date", "level_1": "symbol", 0: "chaikin_oscillator"},
                axis=1,
            )
        )

        for feature_name, feature_df in features_dfs:
            scaler = StandardScaler()
            tmp = pd.DataFrame(
                scaler.fit_transform(feature_df),
                columns=feature_df.columns,
                index=feature_df.index,
            )

            tmp = (
                tmp.stack()
                .reset_index()
                .rename(
                    {"level_0": "date", "level_1": "symbol", 0: feature_name}, axis=1
                )
            )

            df = df.merge(tmp, how="inner", on=["date", "symbol"])

        target = self.create_target(
            close,
            self.days_ahead,
            self.target_type,
            self.q,
        )

        df = df.merge(target, on=["date", "symbol"], how="left")

        return df

    def wavelet_decomposition(self, df: pd.DataFrame):

        haar = pywt.Wavelet("haar")

        features = []
        columns = df.drop((["date", "symbol", "target"]), axis=1).columns
        for name, group in df.groupby("symbol"):
            group_ = group.reset_index(drop=True)
            for c in columns:
                ca, cd1, cd2, cd3, cd4 = pywt.wavedec(
                    group_.loc[:, c].values, haar, level=4, mode="periodization"
                )
                cd1 = np.where(cd1 >= 2.0 * cd1.std(ddof=1, axis=0), 0.0, cd1)
                cd2 = np.where(cd2 >= 2.0 * cd2.std(ddof=1, axis=0), 0.0, cd2)
                cd3 = np.where(cd3 >= 2.0 * cd3.std(ddof=1, axis=0), 0.0, cd3)
                cd4 = np.where(cd4 >= 2.0 * cd4.std(ddof=1, axis=0), 0.0, cd4)
                denoised_signal = pywt.waverec([ca, cd1, cd2, cd3, cd4], haar)
                group_.loc[:, c] = (
                    denoised_signal[1:] if len(group_) % 2 != 0 else denoised_signal
                )
            features.append(group_)

        return pd.concat(features, ignore_index=True).sort_values("date")

    def get_weights(self, df: pd.DataFrame) -> pd.Series:

        output = df.copy()

        output.loc[:, "prob"] = self.grid.predict_proba(
            df.drop(["date", "symbol", "target"], axis=1),
        )[:, -1]

        if self.n_assets:
            output.sort_values("prob", ascending=False, inplace=True)
            output = output.iloc[: self.n_assets]

        output.loc[:, "weight"] = output.loc[:, "prob"] / output.loc[:, "prob"].sum()

        output = pd.Series(
            output.loc[:, "weight"].values, index=output.loc[:, "symbol"].values
        )

        # output = output.loc[:, ["symbol", "weight"]].set_index("symbol")

        return output

    def act(self, observation: Dict[str, pd.Series]) -> pd.Series:

        if len(self.memory_returns) != self.memory_returns.maxlen:

            return self.action_space.sample()

        returns = pd.DataFrame(self.memory_returns).dropna(axis=1)
        close = pd.DataFrame(self.memory_close).loc[:, returns.columns]
        high = pd.DataFrame(self.memory_high).loc[:, returns.columns]
        low = pd.DataFrame(self.memory_low).loc[:, returns.columns]
        volume = pd.DataFrame(self.memory_volume).loc[:, returns.columns]

        if self.screener:

            assets_list = self.screener.filter(returns, volume)
            returns = returns.loc[:, assets_list]
            close = close.loc[:, assets_list]
            high = high.loc[:, assets_list]
            volume = volume.loc[:, assets_list]
            low = low.loc[:, assets_list]

        if (
            self.rebalance_counter % self.rebalance_each_n_obs == 0
            or self.observation_size != returns.shape[1]
        ):
            df = self.preprocess_data(close, low, high, volume)

            if self.reduce_dimensionality:

                p_components = self.pca.fit_transform(
                    df.drop(["date", "symbol", "target"], axis=1)
                )

                df = df.loc[:, ["date", "symbol", "target"]]

                df.loc[:, range(p_components.shape[1])] = p_components

            if self.apply_wavelet_decomp:

                df = self.wavelet_decomposition(df)

            if self.retrain_counter % self.retrain_each_n_obs == 0:

                X = df.dropna(axis=0, subset=["target"]).drop(
                    ["date", "symbol", "target"], axis=1
                )

                y = df.dropna(axis=0, subset=["target"]).loc[:, "target"].astype(int)

                xgb = XGBClassifier(
                    n_jobs=-1,
                    random_state=42,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    n_estimators=250,
                )

                # params = {
                #     "n_estimators": [200, 500, 1000],
                #     "learning_rate": [0.001, 0.01, 0.02],
                #     "reg_alpha": [0.001, 0.1, 1.0, 10.0],
                #     "reg_lambda": [0.001, 0.1, 1.0, 10.0],
                # }

                # self.grid = GridSearchCV(xgb, params, n_jobs=24, cv=3)

                self.grid = xgb

                self.grid.fit(X, y)

            last_obs_date = df.loc[:, "date"].max()

            last_obs = df.loc[df.loc[:, "date"] == last_obs_date]

            w = self.get_weights(last_obs)

            w.name = observation["returns"].name

            if np.sum(w) > 1.0 + self.tol:
                w /= w.sum()

            self.w = w

        self.retrain_counter += 1
        self.rebalance_counter += 1

        return self.w
