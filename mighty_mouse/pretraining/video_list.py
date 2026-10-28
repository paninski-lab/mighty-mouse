"""Flatten the pretraining videos paths file into a list of videos to process.

The videos paths file (`paths.yaml` inside the `pretraining` dir -- not
to be confused with the repo's paths.yaml) is nested `dataset -> split -> video key ->
{path, dims, frames}`. Entries whose `path` hasn't been filled in yet are null.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoEntry:
    dataset: str
    split: str
    key: str
    path: Path
    n_frames: int | None


def list_videos(
    videos_paths: dict,
    dataset: str | None = None,
    splits: tuple[str, ...] | None = None,
) -> tuple[list[VideoEntry], list[tuple[str, str]]]:
    """Flatten a videos paths dict into videos with a usable path.

    Videos are identified by their key in the file, not their file stem: stems aren't
    unique (e.g. every ibl video is named `_iblrig_leftCamera.raw.mp4`).

    Args:
        videos_paths: parsed videos paths file
        dataset: if given, only list this dataset's videos
        splits: if given, only list videos in these splits (e.g. `("train", "unlabeled")`)

    Returns:
        - videos that have a path, across all splits, in file order
        - `(dataset, video key)` of entries skipped because their path is null

    Raises:
        ValueError: if `dataset` isn't in the file, or a key appears in more than one
            split of a dataset (they would write to the same output file)
    """
    if dataset is not None and dataset not in videos_paths:
        raise ValueError(f"unknown dataset {dataset!r}; choices: {sorted(videos_paths)}")

    videos, skipped = [], []
    for name, splits_in_dataset in videos_paths.items():
        if dataset is not None and name != dataset:
            continue
        seen = set()
        for split, videos_in_split in splits_in_dataset.items():
            if splits is not None and split not in splits:
                continue
            for key, info in (videos_in_split or {}).items():
                if not info.get("path"):
                    skipped.append((name, key))
                    continue
                if key in seen:
                    raise ValueError(f"{name}: video key {key!r} appears in more than one split")
                seen.add(key)
                videos.append(VideoEntry(name, split, key, Path(info["path"]), info.get("frames")))
    return videos, skipped
