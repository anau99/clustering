import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


def max_abs_scaler(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Scale numeric features by their maximum absolute value.
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["number"]).columns.tolist()

    if len(columns) == 0:
        return result

    scaler = MaxAbsScaler()

    result[columns] = scaler.fit_transform(result[columns])

    return result