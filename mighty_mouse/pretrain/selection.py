"""Stage 2 of frame selection, step 2: choose a fixed number of frames inside one dataset.

Two methods, both working on the dataset's stage-1 pool of unit-length embeddings:

- `hier`: balanced two-level k-means (`hierarchical.hierarchical_select`), which under-weights
  dense stretches and keeps rare poses without chasing outliers;
- `fps`: greedy farthest-point sampling (`farthest_point.farthest_point_select`), the most
  aggressive about rare poses and outliers, at a cost in typical-frame coverage.

torch is imported inside the functions that need it.
"""

import numpy as np

METHODS = ("hier", "fps")
K1_FACTOR = 3          # hier: level-1 clusters as a multiple of the number of frames wanted
MAX_K1_SHARE = 0.9     # hier: level-1 clusters as a share of the pool, at most
MIN_K2 = 50            # hier: fewest level-2 groups
K2_DIVISOR = 30        # hier: level-2 groups = level-1 clusters / K2_DIVISOR


def hier_sizes(n: int, n_select: int) -> tuple[int, int]:
    """`(k1, k2)` for balanced two-level selection of `n_select` of `n` rows.

    `k1` is `K1_FACTOR * n_select` limited to `MAX_K1_SHARE` of the rows, but always more than
    `n_select` and at most `n`.

    Raises:
        ValueError: if `n_select` is not in [1, n)
    """
    if not 1 <= n_select < n:
        raise ValueError(f"n_select must be in [1, {n}), got {n_select}")
    k1 = min(n, max(n_select + 1, min(K1_FACTOR * n_select, int(MAX_K1_SHARE * n))))
    return k1, max(MIN_K2, k1 // K2_DIVISOR)


def select_pool(x, n_select: int, method: str, seed: int) -> np.ndarray:
    """Choose `n_select` rows of a dataset's pool.

    Args:
        x: unit-length rows, shape (n, d), on the device to run on
        n_select: number of rows wanted; all rows are returned if it is at least n
        method: one of `METHODS`
        seed: seed for clustering / the first farthest-point pick

    Returns:
        ascending row indices. `hier` may return fewer than `n_select` if k-means leaves
        level-1 clusters empty.

    Raises:
        ValueError: if `method` is unknown or `n_select` is not positive
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choices: {METHODS}")
    if n_select < 1:
        raise ValueError(f"n_select must be positive, got {n_select}")
    n = len(x)
    if n_select >= n:
        return np.arange(n)
    if method == "fps":
        from mighty_mouse.pretrain.farthest_point import farthest_point_select

        return farthest_point_select(x, n_select, seed)[0]
    from mighty_mouse.pretrain.hierarchical import hierarchical_select

    k1, k2 = hier_sizes(n, n_select)
    return hierarchical_select(x, n_select, k1, k2, seed)[0]
