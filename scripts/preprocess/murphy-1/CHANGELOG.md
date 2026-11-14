# murphy-1 dataset changelog

Mouse reaching/paw recordings from three cameras (cama/camb/camc), converted by
[`convert_murphy_1.py`](convert_murphy_1.py) (see [`README.md`](README.md)). Keypoint
mapping: [`configs/datasets/murphy-1.yaml`](../../../configs/datasets/murphy-1.yaml).

## Changelog

### 2026-11-14 (MW) (version 1)
- Dropped the `snout_A` and `mouth_tip_A` keypoints; the schema is now the 8 digit tips
  (`r{1-4}_tip_A`, `l{1-4}_tip_A`).
- Removed every frame where either paw has 1 or 2 digit tips labeled, keeping only frames with
  0, 3 or 4 labeled tips per paw (a tip counts as labeled when both x and y are present). This
  removed 1917 of 5907 frames, leaving 3990.
- Re-split at the subject level on the filtered frames: 3480 train / 510 test (12.8%). The test
  subjects differ from v0 (now `outliers_camc`, `test_Trial3_camb`, `FS`, `G8S`, `U`). No label
  values were edited.

### 2026-11-11 (MW) (version 0)
- Initial conversion from `_raw/_dlc/murphy-1` (cama/camb/camc merged): 5907 frames (5160 train /
  747 test) after removing 52 byte-identical duplicate frames; no label edits.
- Keypoints (10): `r{1-4}_tip_A`, `l{1-4}_tip_A`, `snout_A`, `mouth_tip_A`. Dropped from the source:
  `*_mid_A`, `*_joint_A`, `r_mid_A`, `l_mid_A`, `spout_tip_A`.
