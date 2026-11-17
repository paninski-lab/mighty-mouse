# aind-foraging-bot dataset changelog

## Changelog

### 2026-11-17 (MW) (version 0)
- Initial conversion from `_raw/_dlc/aind-forage/Foraging_Bot_UpdatingOld_20220927-LucasKinsey-2022-09-27`: 723 frames (622 train / 101 test, 35 / 4 videos split at the video level, seed 4); no label edits.
- Keypoints (7): `nose_tip`, `tongue_tip`, `tongue_end_left`, `tongue_end_right`, `lowerlip`, `paw_wrist_left`, `paw_wrist_right`. Left/right are the animal's side (bottom-up camera, so swapped from the source's image-side `L`/`R`). Dropped from the source: lickports, back tongue points, whiskers.
