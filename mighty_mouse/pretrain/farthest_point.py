"""Greedy farthest-point selection (an epsilon-net) of rows by cosine distance.

Start from a random row and repeatedly add the row farthest from everything chosen so far.
Dense regions are thinned first and isolated rows are kept, so rare poses survive but so do
outliers (including junk); the typical-frame coverage is worse than with k-means.

torch is imported inside the function.
"""

import numpy as np


def farthest_point_select(x, n_select: int, seed: int) -> tuple[np.ndarray, dict]:
    """Pick `n_select` rows of `x` by greedy farthest-point sampling.

    Args:
        x: unit-length rows, shape (n, d), on the device to run on
        n_select: number of rows to pick, at most n
        seed: seed for the first row

    Returns:
        - indices of the picked rows, ascending
        - stats: `eps_last`, the cosine distance from the farthest unpicked row to its nearest
          picked one after the last pick (the net's radius; 0 if everything was picked)

    Raises:
        ValueError: if `n_select` is not in [1, n]
    """
    import torch

    n = len(x)
    if not 1 <= n_select <= n:
        raise ValueError(f"n_select must be in [1, {n}], got {n_select}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = torch.empty(n_select, dtype=torch.int64, device=x.device)
    chosen[0] = int(torch.randint(n, (1,), generator=generator))
    min_dist = torch.full((n,), 2.0, dtype=torch.float32, device=x.device)
    for i in range(1, n_select):
        min_dist = torch.minimum(min_dist, 1 - (x @ x[chosen[i - 1]]).float())
        min_dist[chosen[i - 1]] = -1.0  # picked rows are never the farthest again
        chosen[i] = torch.argmax(min_dist)
    min_dist = torch.minimum(min_dist, 1 - (x @ x[chosen[-1]]).float())
    min_dist[chosen] = -1.0
    return np.sort(chosen.cpu().numpy()), {"eps_last": max(float(min_dist.max()), 0.0)}
