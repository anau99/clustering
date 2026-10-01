import hashlib

import pandas as pd


def hash_encoder(df: pd.DataFrame, columns=None, n_features=8) -> pd.DataFrame:
    """
    Hash categorical columns into fixed-size integer features.
    """

    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:

        hashed = result[col].astype(str).apply(
            lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16)
        )

        result[col] = hashed % n_features

    return result