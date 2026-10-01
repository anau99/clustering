import pandas as pd
from sklearn.preprocessing import StandardScaler


def standard_scaler(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Standardize selected columns.
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["number"]).columns.tolist()

    if len(columns) == 0:
        return result

    scaler = StandardScaler()
    result[columns] = scaler.fit_transform(result[columns])

    return result