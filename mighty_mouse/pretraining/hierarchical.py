"""Two-level k-means selection of a fixed number of frames, balanced across the data's
broad groups rather than proportional to density.

Plain k-means places clusters in proportion to density, so a long still stretch of a video
claims most of the clusters. Here many fine (level-1) clusters are grouped into a smaller
number of broad (level-2) groups, and the same number of level-1 clusters is drawn from every
group (groups too small to supply that many give all they have), so each broad group counts
about equally however many frames it holds. See Vo et al. 2024, "Automatic Data Curation for
Self-Supervised Learning: A Clustering-Based Approach".

torch is imported inside the functions that need it.
"""

import numpy as np

from mighty_mouse.pretraining.kmeans import random_member_per_cluster, spherical_kmeans


def balanced_counts(sizes: np.ndarray, target: int) -> int:
    """Per-group quota `n` that fills `target` slots as evenly as possible.

    Groups smaller than `n` give everything they have and the rest give `n`.

    Args:
        sizes: number of items in each group
        target: total number of items wanted

    Returns:
        the largest `n` such that `sum(min(n, size))` is at most `target` (the largest
        group size if all groups together hold at most `target` items)
    """
    lo, hi = 0, int(np.max(sizes))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.minimum(mid, sizes).sum() <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def hierarchical_select(
    x,
    cap: int,
    k1: int,
    k2: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select `cap` rows of `x` by balanced sampling from a two-level k-means.

    1. Level 1: `k1` clusters of the rows of `x`.
    2. Level 2: the level-1 centroids grouped into `k2` groups (capped at the number of
       non-empty level-1 clusters).
    3. `cap` level-1 clusters are chosen: the same quota from every group (`balanced_counts`),
       picked uniformly at random within the group regardless of cluster size, then any
       remaining slots (fewer than the number of groups) filled at random from the
       unchosen clusters.
    4. One random row is kept from each chosen level-1 cluster.

    Args:
        x: unit-length rows, shape (n, d), on the device to run on
        cap: number of rows to select
        k1: number of level-1 clusters; must satisfy `cap < k1 <= n`
        k2: number of level-2 groups
        seed: seed for clustering and all random choices

    Returns:
        - indices of the selected rows, ascending
        - for each, the size of its level-1 cluster
        - stats: `n_iter` (level 1), `n_iter_level2`, `n_empty_clusters` (empty level-1
          clusters)

    Raises:
        ValueError: if `k1` is not in (cap, n]
    """
    import torch

    n = len(x)
    if not cap < k1 <= n:
        raise ValueError(f"k1 must be in ({cap}, {n}], got {k1}")

    centroids1, assign1, n_iter1 = spherical_kmeans(x, k1, seed=seed)
    counts1 = torch.bincount(assign1, minlength=k1)
    ids = (counts1 > 0).nonzero().squeeze(1)            # non-empty level-1 clusters
    stats = {"n_iter": n_iter1, "n_iter_level2": 0, "n_empty_clusters": k1 - len(ids)}

    if len(ids) <= cap:
        chosen_ids = ids
    else:
        k2 = min(k2, len(ids))
        _, assign2, stats["n_iter_level2"] = spherical_kmeans(centroids1[ids], k2, seed=seed)
        group = assign2.cpu().numpy()
        order = np.argsort(group, kind="stable")
        starts = np.searchsorted(group[order], np.arange(k2 + 1))
        sizes = np.diff(starts)
        quota = balanced_counts(sizes[sizes > 0], cap)

        rng = np.random.default_rng(seed)
        chosen = np.concatenate([
            rng.choice(order[starts[j]:starts[j + 1]], min(quota, sizes[j]), replace=False)
            for j in range(k2) if sizes[j] > 0
        ])
        if len(chosen) < cap:
            rest = np.setdiff1d(np.arange(len(ids)), chosen)
            chosen = np.concatenate([chosen, rng.choice(rest, cap - len(chosen), replace=False)])
        chosen_ids = ids[torch.from_numpy(chosen).to(ids.device)]

    is_chosen = torch.zeros(k1, dtype=torch.bool, device=x.device)
    is_chosen[chosen_ids] = True
    members = random_member_per_cluster(assign1, k1, seed)
    selected = members[is_chosen[assign1[members]]]
    return selected.cpu().numpy(), counts1[assign1[selected]].cpu().numpy(), stats
