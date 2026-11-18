# aind-foraging-bot dataset changelog

## Changelog

### 2026-11-17 (MW) (version 1)
- Added the `nose_bottom` keypoint (appended last in `keypoint_names`) and labeled it by hand: 621 of 622 train frames and 101 of 101 test frames.
- Relabeled paws: `paw_wrist_left` changed on 145 train / 2 test frames, `paw_wrist_right` on 129 train / 24 test frames. Some paw labels were removed (now NaN): `paw_wrist_right` 546 -> 537 train and 101 -> 90 test labeled frames, `paw_wrist_left` 55 -> 54 labeled test frames.
- Small fixes: `lowerlip` y in `bottom_face_1-0000/img0460.png` (124.5 -> 109.8, was too far from `nose_tip`); tongue points removed in `bottom_face_226-0000/img1838.png`.
- Header fix: added the missing `scorer`/`bodyparts`/`coords` labels to the first header cells of the CSVs (including the version0 snapshots); no label values changed by this.

### 2026-11-17 (MW) (version 0)
- Initial conversion from `_raw/_dlc/aind-forage/Foraging_Bot_UpdatingOld_20220927-LucasKinsey-2022-09-27`: 723 frames (622 train / 101 test, 35 / 4 videos split at the video level, seed 4); no label edits.
- Keypoints (7): `nose_tip`, `tongue_tip`, `tongue_end_left`, `tongue_end_right`, `lowerlip`, `paw_wrist_left`, `paw_wrist_right`. Left/right are the animal's side (bottom-up camera, so swapped from the source's image-side `L`/`R`). Dropped from the source: lickports, back tongue points, whiskers.
