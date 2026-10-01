import pandas as pd


def frequency_encoder(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Replace categories with their frequencies.
    """

    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:
        freq = result[col].value_counts(normalize=True)
        result[col] = result[col].map(freq)

    return result