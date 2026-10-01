import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoder(df: pd.DataFrame,columns=None,encode_all=False) -> pd.DataFrame:

    result = df.copy()
    if columns is not None:
        pass

    elif encode_all:
        columns = result.columns.tolist()
    else:
        columns = result.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    if len(columns) == 0:
        return result

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = encoder.fit_transform(result[columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(columns),
        index=result.index
    )

    result = result.drop(columns=columns)
    result = pd.concat([result, encoded_df], axis=1)

    return result