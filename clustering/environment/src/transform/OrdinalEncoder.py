import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def ordinal_encoder(df: pd.DataFrame, columns=None, encode_all=False) -> pd.DataFrame:
    """
    Encode selected categorical columns using OrdinalEncoder.
    """
    result = df.copy()

    if columns is not None:
        pass

    elif encode_all:
        columns = result.columns.tolist()
    else:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(columns) == 0:
        return result

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    result[columns] = encoder.fit_transform(result[columns])

    return result