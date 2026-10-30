# aind-vbn-face dataset changelog

See [`README.md`](README.md) for the conversion pipeline (shared with
[`aind-vbn-body`](../aind-vbn-body/README.md)).

## Changelog

### 2026-10-20 (MW) (version 1)
- Removed every column except `tongue_tip` (its labels are kept) and rebuilt the label set in
  labeling order: `eye_{front,back,top,bottom}_l`, `whisker_pad_l_{top,side}`,
  `eye_{front,back,top,bottom}_r`, `whisker_pad_r_{top,side}`, `nose_top`, `nose_tip`,
  `tongue_tip`, `paw_wrist_l`, `paw_wrist_r`. Dropped: `eye_top_l`, `eye_bottom_l`, `ear_tip_l`,
  `ear_base_l`, `nostril_l`/`nostril_r`, `jaw`, `tongue_base_l`/`tongue_base_r`,
  `paw_forward_l`/`paw_forward_r` and the four finger-tip columns of each paw. The old
  `whisker_pad_l_*`, `nose_tip` and `paw_wrist_*` labels were replaced by new ones.
- Pseudo-labeled the 16 keypoints other than `tongue_tip` in both CSVs with the
  `head-fixed_v9/nonlinear_head` model (confidence >= 0.9; cells below threshold left empty).
- Then corrected them by active pseudo-labeling with 3-model standalone ensembles, in four
  rounds of hand-reviewed rows in the labeling app (253 rows): round 0, 136 rows (the 97 rows
  of the first three sessions, plus 3 frames from each of the 13 other sessions); round 1, +39
  (3 highest-disagreement frames per session); round 2, +52 (4 per session); round 3, +26 (every
  frame with a wrist spread > 3 px, plus the 20 with the largest `nose_top` spread).
- Filled the other 310 rows with the median of the ensemble members with confidence >= 0.9
  (empty where none cleared it). The first row of each CSV (`img017140.png` in train,
  `img006760.png` in test) was not pseudo-labeled and was labeled by hand. Some `tongue_tip`
  labels and later spot corrections were edited by hand; those edits aren't itemized.

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
