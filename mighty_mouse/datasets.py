"""
Single source of truth for which datasets are registered in the combined corpus.

Used by both scripts/build_dataset.py (as the default --datasets set) and
mighty_mouse/train.py (as the set evaluated against after each training run) --
previously two separately-maintained copies of this list that had to be kept
in sync by hand. This does NOT derive from configs/datasets/*.yaml; adding a
dataset here is still a manual step (see skills/preprocess-new-dataset/SKILL.md,
stage 2) since not every configs/datasets/<name>.yaml is necessarily ready to
be part of default runs yet.
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
    "kondo",
    "tyssowski",
]
