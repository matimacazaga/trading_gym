import pandas as pd
import numpy as np


def clean(df: pd.DataFrame) -> pd.DataFrame:

    df = df.replace([np.inf, -np.inf], np.nan)

    return df.dropna()


def align_index(source: pd.DataFrame, target: pd.DataFrame):

    return target.loc[source.index, :].copy()
