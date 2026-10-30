"""Shared fixtures for the labeling tests: tiny label frames and fake model predictions."""

import numpy as np
import pandas as pd
import pytest

SCORER = "All"
KPS = ["nose", "ear_a", "ear_b"]
EARS = ["ear_a", "ear_b"]


def label_frame(images: list[str], nose: float = 1.0) -> pd.DataFrame:
    """Labels for `images`: nose labeled, both ears empty."""
    cols = pd.MultiIndex.from_tuples(
        [(SCORER, kp, c) for kp in KPS for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(np.nan, index=pd.Index(images), columns=cols)
    df[(SCORER, "nose", "x")] = nose
    df[(SCORER, "nose", "y")] = nose + 1
    return df


def pred_frame(
    images: list[str], values: dict[str, list[tuple[float, float, float]]],
) -> pd.DataFrame:
    """Predictions: {kp: [(x, y, likelihood) per image]}."""
    cols, data = [], []
    for kp, vals in values.items():
        arr = np.array(vals, dtype=float)
        for j, c in enumerate(("x", "y", "likelihood")):
            cols.append(("model", kp, c))
            data.append(arr[:, j])
    return pd.DataFrame(
        np.stack(data, axis=1), index=pd.Index(images),
        columns=pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"]),
    )


@pytest.fixture
def train_df() -> pd.DataFrame:
    return label_frame([f"labeled-data/s{s}/img{i}.png" for s in "AB" for i in range(4)])


@pytest.fixture
def test_df() -> pd.DataFrame:
    return label_frame([f"labeled-data/sC/img{i}.png" for i in range(2)], nose=5.0)
