# aind-vbn-body dataset changelog

See [`README.md`](README.md) for the conversion pipeline.

## Changelog

### 2026-09-29 (MW) (version 0)
- Initial conversion from the `.behavior` projects in `_raw/_dlc/aind-vbn` via
  `convert_aind_vbn.py --view body`: 563 frames (489 train / 74 test, 16 sessions, 16 mice,
  2 test mice), plus 15 s clips of each session video.
- Cropped each frame to face + upper trunk (box from `nose_tip`/eye keypoints, see
  [`README.md`](README.md)) and resized to 256x256. The 15 s video clips are cropped the same
  way, using each session's median box over its labeled frames.
- Dropped `*_lh`, `*_rh` and `tail_*` keypoints (40 -> 25).
- Drafted `configs/datasets/aind-vbn-body.yaml` (8 keypoints mapped, 17 excluded, all
  sessions `left`). Stage 2 not started.
