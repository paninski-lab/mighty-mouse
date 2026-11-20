"""Stage 2 of frame selection, step 1: how many frames each dataset gets.

A dataset's share is proportional to its frame count raised to a power `alpha` below 1, so big
datasets are oversampled relative to small ones but not at their raw share, and a dataset can
never give more frames than stage 1 kept for it.
"""

import numpy as np

DEFAULT_ALPHA = 0.3


def dataset_budgets(
    raw_frames: dict[str, int],
    pool_frames: dict[str, int],
    total: int,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, int]:
    """Split `total` frames over datasets in proportion to `raw_frames ** alpha`.

    A dataset whose share exceeds its pool (the frames stage 1 kept) gets its whole pool and
    the surplus is redistributed over the others in the same proportions. Integer budgets are
    rounded by largest remainder so they sum to exactly `total`.

    Args:
        raw_frames: dataset -> number of frames in its videos
        pool_frames: dataset -> number of frames stage 1 kept (the cap); same keys
        total: frames wanted overall
        alpha: exponent; 0 gives every dataset the same share, 1 shares by raw frame count

    Returns:
        dataset -> budget, at most its pool size. If the pools together hold at most `total`
        frames, every dataset gets its whole pool.

    Raises:
        ValueError: if the keys differ, `total` is not positive, `alpha` is negative, or a
            dataset has no frames
    """
    if raw_frames.keys() != pool_frames.keys():
        raise ValueError("raw_frames and pool_frames must have the same datasets")
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    names = sorted(raw_frames)
    raw = np.array([raw_frames[d] for d in names], dtype=float)
    pool = np.array([pool_frames[d] for d in names], dtype=float)
    if (raw <= 0).any() or (pool <= 0).any():
        empty = [d for d, r, p in zip(names, raw, pool, strict=True) if r <= 0 or p <= 0]
        raise ValueError(f"datasets without frames: {empty}")

    if pool.sum() <= total:
        return {d: int(p) for d, p in zip(names, pool, strict=True)}

    weights = raw**alpha
    budget = np.zeros(len(names))
    free = np.ones(len(names), dtype=bool)
    left = float(total)
    while True:  # cap datasets at their pool and share what's left among the rest
        budget[free] = left * weights[free] / weights[free].sum()
        over = free & (budget > pool)
        if not over.any():
            break
        budget[over] = pool[over]
        left -= pool[over].sum()
        free &= ~over

    rounded = np.floor(budget).astype(int)
    short = total - rounded.sum()
    room = np.flatnonzero(rounded < pool)
    for i in room[np.argsort(-(budget - rounded)[room])][:short]:
        rounded[i] += 1
    return {d: int(b) for d, b in zip(names, rounded, strict=True)}
