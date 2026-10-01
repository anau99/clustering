import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def kbins_discretizer(df: pd.DataFrame,columns=None, n_bins=5, strategy="quantile") -> pd.DataFrame:
    """
    Discretize numeric columns into bins.
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["number"]).columns.tolist()

    if len(columns) == 0:
        return result

    encoder = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=strategy
    )

    result[columns] = encoder.fit_transform(result[columns])

    return result