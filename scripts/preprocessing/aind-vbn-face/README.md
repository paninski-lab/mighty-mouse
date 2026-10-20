# aind-vbn-face dataset conversion

> **Status:** `aind-vbn-face` is in `ALL_DATASETS` (`mighty_mouse/datasets.py`); `convert_dataset.py`
> and `build_dataset.py` haven't been run for it yet. The keypoint mapping is in
> [`configs/datasets/aind-vbn-face.yaml`](../../../configs/datasets/aind-vbn-face.yaml).

Built from the `.face` projects in `_raw/_dlc/aind-vbn` (front-facing face camera, 658x492) by
the same script as the body dataset, `scripts/preprocessing/aind-vbn-body/convert_aind_vbn.py
--view face`. See [`../aind-vbn-body/README.md`](../aind-vbn-body/README.md) for the source
format, train/test split (same held-out mice as body), keypoint dropping, video clips and
everything else shared between the two.

## Face-specific decisions

- **Crop**: a fixed 300x300 px box relative to `nose_tip`, x in [nose-120, nose+180] and
  y in [nose-200, nose+100], resized to 256x256. Frames missing `nose_tip` (4 of 564) use the
  session's median nose position. Video clips use the session's median box.
- **Keypoints**: the conversion script produces the source schema, but the live labels have
  since been overhauled (see [`CHANGELOG.md`](CHANGELOG.md)): all keypoints except `tongue_tip`
  were removed, and the eyes, whisker pads, nose and wrists are pseudo-labeled for both sides.
  The config maps them to their `_left`/`_right` names one-to-one, not with a `{side}` template,
  because both sides appear in the same frame; the sessions are all listed as `left` only
  because the config validation requires every session to appear.

## Running

```bash
python scripts/preprocessing/aind-vbn-body/convert_aind_vbn.py --view face
```

Spot-checked by overlaying keypoints on 6 cropped frames and eyeballing a clip frame.
