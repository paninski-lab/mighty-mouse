# murphy-1 dataset changelog

Mouse reaching/paw recordings from three cameras (cama/camb/camc), converted by
[`convert_murphy_1.py`](convert_murphy_1.py) (see [`README.md`](README.md)). Keypoint
mapping: [`configs/datasets/murphy-1.yaml`](../../../configs/datasets/murphy-1.yaml)
(untracked for now).

## Changelog

### 2026-11-11 (MW) (version 0)
- Initial conversion from `_raw/_dlc/murphy-1` (cama/camb/camc merged): 5907 frames (5160 train /
  747 test) after removing 52 byte-identical duplicate frames; no label edits.
- Keypoints (10): `r{1-4}_tip_A`, `l{1-4}_tip_A`, `snout_A`, `mouth_tip_A`. Dropped from the source:
  `*_mid_A`, `*_joint_A`, `r_mid_A`, `l_mid_A`, `spout_tip_A`.
