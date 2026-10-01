import pandas as pd
from sklearn.preprocessing import Normalizer


def normalizer(df: pd.DataFrame, columns=None, norm="l2") -> pd.DataFrame:
    """
    Normalize samples using L1, L2 or Max norm.
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["number"]).columns.tolist()

    if len(columns) == 0:
        return result

    scaler = Normalizer(norm=norm)

    result[columns] = scaler.fit_transform(result[columns])

    return result