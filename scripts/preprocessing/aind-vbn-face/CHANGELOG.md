# aind-vbn-face dataset changelog

See [`README.md`](README.md) for the conversion pipeline (shared with
[`aind-vbn-body`](../aind-vbn-body/README.md)).

## Changelog

### 2026-09-29 (MW) (version 0)
- Initial conversion from the `.face` projects in `_raw/_dlc/aind-vbn` via
  `convert_aind_vbn.py --view face`: 564 frames (490 train / 74 test, 16 sessions, same 2
  test mice as `aind-vbn-body`), plus 15 s clips of each session video.
- Cropped each frame to a 300x300 box around `nose_tip` (see [`README.md`](README.md)) and
  resized to 256x256; video clips cropped the same way with each session's median box.
- Dropped `*_lh`, `*_rh` and `tail_*` keypoints (40 -> 25).
- Drafted `configs/datasets/aind-vbn-face.yaml` (16 keypoints mapped, including finger tips for
  both sides; 9 excluded incl. all `paw_*`; eye/pad/ear mapped directly to `_left`). Stage 2 not started.
- Added `POST_PROCESS["aind-vbn-face"]` (`mighty_mouse/convert.py`) forcing the all-missing
  `eye_*_right`/`pad_*_right`/`ear_*_right` columns to `visible=0`.
