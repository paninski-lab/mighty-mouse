"""
Camera view of a labeled frame, from its session directory name. Renderers draw one panel (or
column) per view and sample frames per view.
Session names are the same in a raw dataset and in the converted corpus.
"""

import re
from collections.abc import Iterable

import numpy as np

from mighty_mouse.labels import session_name

# datasets recorded from one camera
SINGLE_VIEW = {"cazettes-side": "side", "kondo": "bottom",
               "aind-vbn-body": "body", "aind-vbn-face": "face"}


def view_of(dataset: str, session: str) -> str:
    """Camera view of one session of a dataset.

    Datasets without a rule of their own fall back to a `cam<N>` or `_left` / `_right` suffix
    of the session name, then to "camera" (one view).

    Args:
        dataset: dataset name (an `ibl*` name covers ibl and sets derived from it)
        session: session directory name (labeled-data/<session>/ or
            labeled-data/<dataset>/<session>/)

    Returns:
        view name, e.g. "left", "cam1", "BC"
    """
    if dataset.startswith("ibl"):
        # <uuid>_left / <uuid>_right_flipped, or legacy <...>_leftCamera.<...> (left, unflipped)
        left = session.endswith("_left") or ("leftCamera" in session and "right" not in session)
        return "left" if left else "right"
    if dataset == "facemap":
        return "cam0" if "cam0" in session else "cam1"
    if dataset.startswith("cheese"):                 # six cameras, named in the session
        m = re.search(r"_(BC|TC|TL|TR|L|R)(?:_|$)", session)
        return m.group(1) if m else "camera"
    if dataset in ("hantman-mv", "tyssowski"):
        return "front" if session.endswith("_front") else "side"
    if dataset in SINGLE_VIEW:
        return SINGLE_VIEW[dataset]
    m = re.search(r"(cam\d+)$", session)             # e.g. kaufman <session>-cam1 / -cam2
    if m:
        return m.group(1)
    m = re.search(r"_(left|right)$", session)        # e.g. cavelli <session>_left / _right
    return m.group(1) if m else "camera"


def sample_per_view(frames: Iterable[str], dataset: str, n: int) -> dict[str, list[str]]:
    """At most `n` frames per camera view, evenly spaced through each view's frames (so across
    sessions), in the order given.

    Args:
        frames: frame image paths ending in <session>/<frame>
        dataset: dataset name, to tell the views apart (see `view_of`)
        n: frames kept per view

    Returns:
        {view: frames}, views sorted by name
    """
    by_view: dict[str, list[str]] = {}
    for frame in frames:
        by_view.setdefault(view_of(dataset, session_name(frame)), []).append(frame)
    return {view: [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(n, len(fs)))]
            for view, fs in sorted(by_view.items())}
