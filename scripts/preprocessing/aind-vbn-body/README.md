# aind-vbn-body dataset conversion

> **Status: stage 1 only.** `_raw/aind-vbn-body/` is a usable standalone LP project, not yet in
> the combined corpus. See [`skills/preprocess-new-dataset/SKILL.md`](../../../skills/preprocess-new-dataset/SKILL.md) for what
> stage 2 would involve; don't start it unless asked. A draft keypoint mapping is in
> [`configs/datasets/aind-vbn-body.yaml`](../../../configs/datasets/aind-vbn-body.yaml).

## Source format

`_raw/_dlc/aind-vbn/` holds one standard DLC project per (session, camera): 16 sessions x
{`.behavior`, `.face`}, e.g. `1044385384_524761_20200819.behavior-Corbett-2023-06-29`
(`<session_id>_<mouse_id>_<date>.<camera>-<scorer>-<label date>`). Each has one
`labeled-data/<session>.<camera>/` with `CollectedData_<scorer>.csv` and a ~2.7 h, 60 fps video.

The `.behavior` projects are the **body** camera (side view, 658x492) and are what this
dataset is built from; the `.face` projects become the parallel `aind-vbn-face` dataset
using the same script (`--view face`).

## Decisions

- **Two parallel datasets**, not merged: `aind-vbn-body` / `aind-vbn-face`, one script with `--view`.
- **Crop (body only)**: each frame is cropped to face + upper trunk and resized to 256x256.
  H = |`nose_tip`.x - eye_mid.x| (eye_mid = mean of `eye_top_l`/`eye_bottom_l`), C = mean of
  `nose_tip` and eye_mid; box = [C-2H, C+3H] in x and y (square, ~5H, ~165 px in the source).
  Frames missing nose/eyes (5 of 563) use the session's median H and C. Keypoints are
  shifted/scaled with the crop and set to NaN if they fall outside it. Video clips use each
  session's median box (median H, C over its labeled frames) instead of per-frame boxes,
  also resized to 256x256.
- **Keypoints**: `*_lh`, `*_rh` (hindpaws) and `tail_*` are dropped entirely, leaving 25 with
  their raw source names (see `project.yaml`). The config maps 8 of them to canonical names
  and excludes the other 17. Every source name is left-side, so all sessions are `left`.
  The scorer varies by annotator, so it's normalized to `aind-vbn` when concatenating.
  Session `1051155866_…behavior` has a stale default `config.yaml`; CSV headers are used instead.
- **Split**: subject-level by mouse id via `mighty_mouse.subject_split` (target 12.5% midpoint,
  seed 0). Test mice are always derived from the body frame counts, so the face dataset
  holds out the same mice.
- **Videos**: 15 s clip per session, taken from the start after skipping 60 s (no motion
  search on ~2.7 h videos). Test-mouse sessions go to `videos_test/`, the rest to `videos/`.
- **Ignored**: `old/`, `time_stamps.csv`, `vbn_video_metadata.csv`.

## Running

```bash
python scripts/preprocessing/aind-vbn-body/convert_aind_vbn.py --view body
```

Spot-checked by overlaying keypoints on 6 cropped frames (and eyeballing a frame from two clips); they land on the right body parts.
