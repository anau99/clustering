import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def polynomial_features(df: pd.DataFrame, columns=None, degree=2, include_bias=False) -> pd.DataFrame:
    """
    Generate polynomial feature combinations.
    """

    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=["number"]).columns.tolist()

    if len(columns) == 0:
        return result

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias
    )

    transformed = poly.fit_transform(result[columns])

    poly_df = pd.DataFrame(
        transformed,
        columns=poly.get_feature_names_out(columns),
        index=result.index
    )

    result = result.drop(columns=columns)
    result = pd.concat([result, poly_df], axis=1)

    return result