"""Chunked spherical k-means (cosine distance) for embeddings that fit on one device.

Inputs are unit-length vectors, so "closest centroid" is `argmax x . c` and centroids are
re-normalized after every update. The (n, k) similarity matrix is never held at once: it
is computed `chunk` rows at a time. torch is imported inside the functions so this
module can be imported without it.
"""

DEFAULT_N_ITER = 50
DEFAULT_TOL = 1e-3
DEFAULT_CHUNK = 32768


def normalize_rows(x, chunk: int = DEFAULT_CHUNK):
    """L2-normalize the rows of `x` in place (fp32 math per chunk, stored in x's dtype).

    Args:
        x: tensor, shape (n, d)
        chunk: rows per fp32 chunk

    Returns:
        `x`
    """
    for start in range(0, len(x), chunk):
        block = x[start:start + chunk].float()
        x[start:start + chunk] = (block / block.norm(dim=1, keepdim=True).clamp_min(1e-6)).to(
            x.dtype
        )
    return x


def assign_to_centroids(x, centroids, chunk: int = DEFAULT_CHUNK):
    """Nearest centroid (highest dot product) for every row of `x`.

    Args:
        x: unit-length rows, shape (n, d)
        centroids: shape (k, d)
        chunk: rows of `x` per similarity block

    Returns:
        - int64 cluster index per row, shape (n,)
        - that row's similarity to its centroid, float32, shape (n,)
    """
    import torch

    assign = torch.empty(len(x), dtype=torch.int64, device=x.device)
    best = torch.empty(len(x), dtype=torch.float32, device=x.device)
    for start in range(0, len(x), chunk):
        sims = x[start:start + chunk].float() @ centroids.T
        best[start:start + chunk], assign[start:start + chunk] = sims.max(dim=1)
    return assign, best


def kmeans_pp_init(x, k: int, generator):
    """k-means++ seeding with cosine distance (`1 - x . c`, proportional to squared
    Euclidean distance for unit vectors).

    Args:
        x: unit-length rows, shape (n, d)
        k: number of centroids, at most n
        generator: `torch.Generator` on `x`'s device

    Returns:
        float32 centroids, shape (k, d)
    """
    import torch

    n = len(x)
    chosen = torch.empty(k, dtype=torch.int64, device=x.device)
    chosen[0] = torch.randint(n, (1,), generator=generator, device=x.device)[0]
    min_dist = (1 - x @ x[chosen[0]]).float().clamp_min(0)
    for i in range(1, k):
        chosen[i] = torch.multinomial(min_dist + 1e-12, 1, generator=generator)[0]
        min_dist = torch.minimum(min_dist, (1 - x @ x[chosen[i]]).float().clamp_min(0))
    return x[chosen].float()


def spherical_kmeans(
    x,
    k: int,
    seed: int = 0,
    n_iter: int = DEFAULT_N_ITER,
    tol: float = DEFAULT_TOL,
    chunk: int = DEFAULT_CHUNK,
    init: str = "kmeans++",
):
    """Cluster unit-length rows of `x` into `k` clusters by cosine similarity.

    Empty clusters are re-seeded with the points farthest from their centroids. Stops
    after `n_iter` iterations, or once fewer than `tol` of the rows change cluster.

    Args:
        x: unit-length rows, shape (n, d), on the device to run on
        k: number of clusters, at most n
        seed: seed for initialization
        n_iter: maximum iterations
        tol: stop when this fraction of rows (or less) changed cluster in an iteration
        chunk: rows per similarity block
        init: `"kmeans++"` or `"random"`

    Returns:
        - unit-length centroids, float32, shape (k, d)
        - int64 cluster index per row, shape (n,), consistent with the centroids
        - number of iterations run

    Raises:
        ValueError: if `k` is not in [1, n] or `init` is unknown
    """
    import torch

    n = len(x)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if init not in ("kmeans++", "random"):
        raise ValueError(f"unknown init {init!r}")

    generator = torch.Generator(device=x.device).manual_seed(seed)
    if init == "kmeans++":
        centroids = kmeans_pp_init(x, k, generator)
    else:
        centroids = x[torch.randperm(n, generator=generator, device=x.device)[:k]].float()

    prev = None
    converged = False
    n_run = 0
    for it in range(1, n_iter + 1):
        n_run = it
        assign, best = assign_to_centroids(x, centroids, chunk)
        if prev is not None and (assign != prev).float().mean() <= tol:
            converged = True
            break
        prev = assign

        sums = torch.zeros_like(centroids)
        for start in range(0, n, chunk):
            sums.index_add_(0, assign[start:start + chunk], x[start:start + chunk].float())
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        if empty.any():
            farthest = best.topk(int(empty.sum()), largest=False).indices
            sums[empty] = x[farthest].float()
        centroids = sums / sums.norm(dim=1, keepdim=True).clamp_min(1e-6)

    if not converged:
        assign, _ = assign_to_centroids(x, centroids, chunk)
    return centroids, assign, n_run
