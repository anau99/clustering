import base64
import gzip
import os

import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


_DIABETES_LABELS_B64 = (
    "H4sIAFnNamoC/zPUMRyFgwga0BzS0j208BHxZhLrA2q5cPCkA/z6DKGYCL0AKNCUQg8EAAA="
)

_HEART_LABELS_B64 = (
    "H4sIAFnNamoC/4WVUQ7DMAxCL7QPuP/l9lNtCTw2TZraOrYTMEQvPT9fT36efP37iOlYpSP2"
    "zfRRy58MqnzWzKiu3oIe37irTlbNHTtyqVPuRZGRHe71ijxenSgnPkL0G3UFCrl/FxvrPL93"
    "x3EFG6pTUA0V16rvN7+u6ePOQsabM6op2KMGAzyfObkezCuUwpPAWZqVNdTR+flGqv7NqkMf"
    "jN5ip1Fu9Xa31BdH8owGVRhci9xpqaR9k1ynMTeo0vC1fUvoBQttduY9L8KV5GOeWmnNqub"
    "G6PB8D2ky76lCzfvgnxL9N4fvTMNUGWeX3FSge6F/tSKEGJO3hQ7e0jsDFwEIAAA="
)


def _decode_labels(b64_str):
    raw = gzip.decompress(base64.b64decode(b64_str))
    return [int(x) for x in raw.decode().split(",")]


DATASETS = [
    {
        "name": "diabetes_data",
        "true_labels": _decode_labels(_DIABETES_LABELS_B64),
        "pred_file": "/app/output_diabetes_data.csv",
        "pred_label_col": "pseudo_label",
        "ari_threshold": 0.34,
        "nmi_threshold": 0.28,
    },
    {
        "name": "heart",
        "true_labels": _decode_labels(_HEART_LABELS_B64),
        "pred_file": "/app/output_heart.csv",
        "pred_label_col": "pseudo_label",
        "ari_threshold": 0.35,
        "nmi_threshold": 0.28,
    },
]


def test_clustering_metrics():
    """
    Compare predicted cluster labels against true labels using
    ARI and NMI for both datasets.
    """

    for cfg in DATASETS:

        y_true = cfg["true_labels"]
        pred_df = pd.read_csv(cfg["pred_file"])
        y_pred = pred_df[cfg["pred_label_col"]].values

        assert len(y_true) == len(y_pred), (
            f"{cfg['name']}: row count mismatch "
            f"(true={len(y_true)}, pred={len(y_pred)})"
        )

        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

        print(
            f"{cfg['name']} -> "
            f"ARI={ari:.4f} | "
            f"NMI={nmi:.4f}"
        )

        assert ari >= cfg["ari_threshold"], (
            f"{cfg['name']}: "
            f"ARI={ari:.4f} < {cfg['ari_threshold']}"
        )

        assert nmi >= cfg["nmi_threshold"], (
            f"{cfg['name']}: "
            f"NMI={nmi:.4f} < {cfg['nmi_threshold']}"
        )


def test_row_count_matches_input():
    """
    Verify that each prediction file contains exactly the same
    number of rows as its corresponding dataset.
    """

    for cfg in DATASETS:

        pred_df = pd.read_csv(cfg["pred_file"])

        assert len(cfg["true_labels"]) == len(pred_df), (
            f"{cfg['name']}: "
            f"input={len(cfg['true_labels'])}, "
            f"prediction={len(pred_df)}"
        )


def test_both_clusters_present():
    """
    Verify that both cluster labels (0 and 1) are present.
    """

    for cfg in DATASETS:

        pred_df = pd.read_csv(cfg["pred_file"])

        labels = set(pred_df[cfg["pred_label_col"]].unique())

        assert labels == {0, 1}, (
            f"{cfg['name']}: "
            f"expected {{0,1}}, found {labels}"
        )


def test_output_single_column():
    """
    Verify that each output file contains exactly one column.
    """

    for cfg in DATASETS:

        pred_df = pd.read_csv(cfg["pred_file"])

        assert pred_df.shape[1] == 1, (
            f"{cfg['name']}: "
            f"expected 1 column, found {pred_df.shape[1]}"
        )


def test_output_file_name():
    """
    Verify that output files exist and have the expected names.
    """

    expected_files = {
        "output_diabetes_data.csv",
        "output_heart.csv",
    }

    for cfg in DATASETS:

        assert os.path.isfile(cfg["pred_file"]), (
            f"File not found: {cfg['pred_file']}"
        )

        assert os.path.basename(cfg["pred_file"]) in expected_files, (
            f"Unexpected file name: "
            f"{os.path.basename(cfg['pred_file'])}"
        )


def test_output_column_name():
    """
    Verify that the only column is named 'pseudo_label'.
    """

    for cfg in DATASETS:

        pred_df = pd.read_csv(cfg["pred_file"])

        assert list(pred_df.columns) == [cfg["pred_label_col"]], (
            f"{cfg['name']}: "
            f"expected column '{cfg['pred_label_col']}', "
            f"found {list(pred_df.columns)}"
        )


if __name__ == "__main__":
    test_clustering_metrics()
    test_row_count_matches_input()
    test_both_clusters_present()
    test_output_single_column()
    test_output_file_name()
    test_output_column_name()