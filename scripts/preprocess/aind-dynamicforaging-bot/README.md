# aind-dynamicforaging-bot dataset conversion

> **Status:** in `ALL_DATASETS` (`mighty_mouse/datasets.py`), with its keypoint mapping in
> [`configs/datasets/aind-dynamicforaging-bot.yaml`](../../../configs/datasets/aind-dynamicforaging-bot.yaml).
> `convert_dataset.py` and `build_dataset.py` haven't been run for it yet.

## Source format

`_raw/_dlc/aind-dynamicforaging/Foraging_Bot_UpdatingOld_20220927-LucasKinsey-2022-09-27`: one standard
DLC project (bottom-up camera on a mouse at a lick spout, 30 fps `.avi` clips), one
`labeled-data/<video>/CollectedData_LucasKinsey.csv` per video, 39 videos / 723 frames. This
is the later superset of `Foraging_Bot-Han_Lucas-2022-04-27` (same 29 videos plus 10 new
ones), so only the September project is used. The CSV header scorer is `Han_Lucas`.

## Decisions

- **Keypoints**: 7 of the source's 17 are kept (names in `RENAME` in the script). Lickports,
  back tongue points and all whisker points are not copied. The camera looks up from below, so
  the source's image-left `L` is the animal's right and left/right are swapped in the output.
- **Images**: native sizes (they differ per video).
- **Split**: whole videos, one video = one subject (no mouse ids available). The default seed is
  the first (seed 4) with 10-15% of frames in test and a video file for every test video -> 4 test
  videos, 101 frames (14.0%).
- **Videos**: for each video present in the source, a 15 s `<video>.short.mp4` (h264, native
  size, 30 fps) taken from the window with the most motion energy (pixel differences), into
  `videos/` (train) or `videos_test/` (test). The source videos are already cropped to ~45-95 s
  so nothing is skipped at the start. 7 labeled videos have no file in the source and so no clip;
  all are in train (their frames are still used). `--videos_only` regenerates just the clips of an
  existing dataset, taking the split from its CSVs.

## After conversion

`nose_bottom` was added afterwards (empty, labeled by hand) with `scripts/preprocess/add_keypoints.py`,
so rerunning the converter yields only the original 7 keypoints and none of the later label edits.

## Running

```bash
python scripts/preprocess/aind-dynamicforaging-bot/convert_aind_dynamicforaging_bot.py [--seed N]
```

Spot-checked by overlaying labels on 6 frames (3 train, 3 test).
