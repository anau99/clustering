import pandas as pd


def binary_encoder(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Binary encode categorical columns.
    """

    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:

        uniques = sorted(result[col].dropna().unique())
        mapping = {v: i for i, v in enumerate(uniques)}

        encoded = result[col].map(mapping).fillna(0).astype(int)

        n_bits = max(1, len(mapping).bit_length())

        binary = encoded.apply(
            lambda x, n_bits=n_bits: list(map(int, format(x, f"0{n_bits}b")))
        )

        binary_df = pd.DataFrame(
            binary.tolist(),
            columns=[f"{col}_bin{i}" for i in range(n_bits)],
            index=result.index
        )

        result.drop(columns=col, inplace=True)
        result = pd.concat([result, binary_df], axis=1)

    return result