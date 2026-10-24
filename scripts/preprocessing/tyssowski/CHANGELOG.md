# tyssowski dataset changelog

Hantman-style reaching task, side + front views. See [`README.md`](README.md) for the
conversion pipeline and [`configs/datasets/tyssowski.yaml`](../../../configs/datasets/tyssowski.yaml)
for the keypoint mapping.

## Changelog

### 2026-10-24 (MW) (version 0)
- Built `_raw/tyssowski` from `_raw/_dlc/tyssowski` with `convert_tyssowski.py` (seed 1):
  174 train / 30 test frames (front and side), 13 videos x 2 views = 26 session dirs, 10
  subjects. Output keeps only `d2_3`, `d3_3`, `d4_3`, `d5_3`, `hand`, `nose`, `mouth`.
  Labels are as received (version 0), no edits.
- Ran `scripts/preprocessing/extract_clips.py` with the same settings as `hantman-mv`'s
  2026-09-22 entry (`--clip-length 8 --skip-start 0 --from-start --fps 500`; the source mp4s
  also carry a wrong `1 fps` tag) on the 13 labeled videos x 2 views: train subjects ->
  `_raw/tyssowski/videos/` (10 videos), test subjects -> `_raw/tyssowski/videos_test/`
  (3 videos). Outputs are 500 fps h264/yuv420p mp4s; 8 s (4000 frames) unless the source
  is shorter (some are only 2000-3500 frames). The 7 unlabeled videos were not clipped.
  Review clips only, not a label change.
