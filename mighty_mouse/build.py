"""
Subsample and merge converted per-dataset CSVs into a combined training set.

The CLI is scripts/build_dataset.py; see docs/build_dataset.md for the design (balanced
vs. full tags, why test sets are never subsampled).
"""

import hashlib

import numpy as np
import pandas as pd


def dataset_rng(seed: int, name: str) -> np.random.Generator:
    """Independent RNG per dataset.

    Seeded from both ``seed`` and the dataset name, so a dataset samples the same frames
    for a given seed regardless of which other datasets are included in a build.
    """
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng([seed, h])


def subsample(df: pd.DataFrame, n_frames: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample up to ``n_frames`` rows without replacement, keeping the original row order.

    Args:
        df: one dataset's converted train split
        n_frames: rows to sample; capped at ``len(df)``. Negative means every row, in
            which case ``rng`` is not used
        rng: see dataset_rng()

    Returns:
        the sampled rows
    """
    if n_frames < 0:
        return df
    idx = rng.choice(len(df), size=min(n_frames, len(df)), replace=False)
    return df.iloc[sorted(idx)]


def build_merged(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-dataset DataFrames. Converted CSVs share identical columns (every
    canonical keypoint, see convert.reindex_to_canonical), so a plain concat suffices."""
    return pd.concat(dfs)
