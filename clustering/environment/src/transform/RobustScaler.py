import pandas as pd
from sklearn.preprocessing import RobustScaler


def robust_scaler(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Scale features using statistics robust to outliers.
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["number"]).columns.tolist()

    if len(columns) == 0:
        return result

    scaler = RobustScaler()
    result[columns] = scaler.fit_transform(result[columns])

    return result