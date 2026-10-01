import pandas as pd


def count_encoder(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Replace categories with occurrence counts.
    """

    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:
        counts = result[col].value_counts()
        result[col] = result[col].map(counts)

    return result