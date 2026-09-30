# aind-vbn-face dataset conversion

> **Status: stage 1 only.** `_raw/aind-vbn-face/` is a usable standalone LP project, not yet in
> the combined corpus. See [`skills/preprocess-new-dataset/SKILL.md`](../../../skills/preprocess-new-dataset/SKILL.md) for what
> stage 2 would involve; don't start it unless asked. A draft keypoint mapping is in
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
- **Keypoints**: same 40-name source schema as body; `*_lh`, `*_rh` and `tail_*` dropped
  (25 left, raw source names; `jaw` is never labeled in this view). The config maps 16
  and excludes the other 9. Unlike body, it keeps the finger labels for both sides
  (`pinky`/`ring_finger`/`middle_finger`/`pointer_finger` -> `d1`-`d4_tip_{left,right}`) and
  excludes all `paw_*` keypoints. The eye, pad and ear keypoints are mapped straight to `_left` names (there is no right-side
  label for them in this dataset); the sessions are all listed as `left` only because the config
  validation requires every session to appear.

**Post-processing:** the `eye_*_right`, `pad_*_right` and `ear_*_right` columns are entirely
missing, so `POST_PROCESS["aind-vbn-face"]` in `mighty_mouse/convert.py` forces them to
`visible=0` (not the default 1). The dataset is still not registered in `ALL_DATASETS`.

## Running

```bash
python scripts/preprocessing/aind-vbn-body/convert_aind_vbn.py --view face
```

Spot-checked by overlaying keypoints on 6 cropped frames and eyeballing a clip frame.
