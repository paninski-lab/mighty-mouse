"""
Single source of truth for which datasets are registered in the combined corpus.

Used by both scripts/preprocess/build_dataset.py (as the default --datasets set) and
mighty_mouse/train/sweep.py (as the set evaluated against after each training run) --
previously two separately-maintained copies of this list that had to be kept
in sync by hand. This does NOT derive from configs/datasets/*.yaml; adding a
dataset here is still a manual step (see skills/preprocess/preprocess-new-dataset/SKILL.md,
stage 2) since not every configs/datasets/<name>.yaml is necessarily ready to
be part of default runs yet.

Invariant: a dataset is in ALL_DATASETS if and only if its configs/datasets/<name>.yaml is
tracked in git (a collaborator's default runs would otherwise fail on the missing config).

SHORT_NAMES gives each dataset the short name used in merged-CSV tags (face+ibl+cheese, see
docs/build_dataset.md); its order is the order names appear in a tag, so a new dataset goes at
the end.
"""

ALL_DATASETS = [
    "aind-vbn-body",
    "aind-vbn-face",
    "cavelli",
    "cazettes-side",
    "cheese-2d",
    "cheese-3d",
    "facemap",
    "hantman-mv",
    "ibl",
    "kaufman",
    "kaufman-water",
    "kondo",
    "tyssowski",
]

# dataset -> short name in merged-CSV tags, in tag order (append new datasets at the end)
SHORT_NAMES = {
    "facemap":       "face",
    "ibl":           "ibl",
    "cheese-2d":     "cheese",
    "cazettes-side": "caz",
    "kondo":         "kondo",
    "hantman-mv":    "hmv",
    "cheese-3d":     "c3d",
    "kaufman":       "kauf",
    "aind-vbn-body": "vbnb",
    "aind-vbn-face": "vbnf",
    "cavelli":       "cav",
    "tyssowski":     "tys",
    "kaufman-water": "kaufw",
}


def tag_of(datasets: list[str]) -> str:
    """Merged-CSV tag of a set of datasets: their short names joined by "+", in SHORT_NAMES order.

    Args:
        datasets: dataset names (order and repeats do not matter)

    Returns:
        e.g. "face+ibl+cheese" for ["cheese-2d", "facemap", "ibl"]

    Raises:
        KeyError: if a dataset has no short name
    """
    unknown = sorted(set(datasets) - set(SHORT_NAMES))
    if unknown:
        raise KeyError(f"no short name for {unknown}: add it to SHORT_NAMES")
    return "+".join(short for name, short in SHORT_NAMES.items() if name in datasets)
