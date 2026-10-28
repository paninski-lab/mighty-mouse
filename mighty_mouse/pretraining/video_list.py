"""Flatten the pretraining videos paths file into a list of videos to process.

The videos paths file (`pretraining_videos_paths_file` in the repo's paths.yaml -- not
to be confused with paths.yaml itself) is nested `dataset -> split -> video key ->
{path, dims, frames}`. Entries whose `path` hasn't been filled in yet are null.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoEntry:
    dataset: str
    stem: str
    path: Path
    n_frames: int | None


def list_videos(
    videos_paths: dict,
    dataset: str | None = None,
) -> tuple[list[VideoEntry], list[tuple[str, str]]]:
    """Flatten a videos paths dict into videos with a usable path.

    Args:
        videos_paths: parsed videos paths file
        dataset: if given, only list this dataset's videos

    Returns:
        - videos that have a path, across all splits, in file order
        - `(dataset, video key)` of entries skipped because their path is null

    Raises:
        ValueError: if `dataset` isn't in the file, or two videos in one dataset share a
            file stem (they would write to the same output file)
    """
    if dataset is not None and dataset not in videos_paths:
        raise ValueError(f"unknown dataset {dataset!r}; choices: {sorted(videos_paths)}")

    videos, skipped = [], []
    for name, splits in videos_paths.items():
        if dataset is not None and name != dataset:
            continue
        seen = {}
        for videos_in_split in splits.values():
            for key, info in (videos_in_split or {}).items():
                if not info.get("path"):
                    skipped.append((name, key))
                    continue
                path = Path(info["path"])
                if path.stem in seen:
                    raise ValueError(
                        f"{name}: videos {seen[path.stem]} and {path} share stem {path.stem!r}"
                    )
                seen[path.stem] = path
                videos.append(VideoEntry(name, path.stem, path, info.get("frames")))
    return videos, skipped
