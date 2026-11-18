# aind-vrforaging-face dataset changelog

See [`README.md`](README.md) for the conversion and pseudo-labeling pipeline.

## Changelog

### 2026-11-18 (MW) (version 0)
- Initial conversion from `_raw/_dlc/aind-vrforaging/face-camera`: 619 frames in 24 sessions (539 train / 80 test, split by mouse, seed 7); the source's 20 labeled frames were discarded and the two `<session> (1)` folders merged into their sessions.
- All labels are pseudo-labels from `head-fixed_v11/mightymouse-v11-384` (confidence >= 0.7; cells below threshold left empty), not hand-checked.
- Keypoints (15): `eye_{back,bottom,front,top}_right`, `nose_tip`, `nose_top`, `nose_bottom`, `pad_{top,side}_right`, `ear_{top,tip,bottom,base}_right`, `wrist_left`, `wrist_right`. Fill is ~100% for eyes, ~97% for nose and pads, 40-80% for ears and wrists.
