# hou-tube dataset changelog

Six-camera (L/R/TL/TR/BC/TC) mouse orofacial recordings, hand-labeled; converted by
[`convert_hou_tube.py`](convert_hou_tube.py) (see [`README.md`](README.md)). Keypoint
mapping: [`configs/datasets/hou-tube.yaml`](../../../configs/datasets/hou-tube.yaml)
(untracked for now).

## Changelog

### 2026-10-26 (MW)
- Initial conversion from `_raw/_dlc/hou-tube` into `_raw/hou-tube`; held out one recording
  for test. Not registered in `ALL_DATASETS`; config gitignored. Not versioned yet (`bump_version.py` not run).
