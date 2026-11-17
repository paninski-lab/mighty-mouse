# aind-foraging-bot dataset conversion

> **Status:** converted to `_raw/aind-foraging-bot/` (LP format) only. Not yet in `ALL_DATASETS`
> (`mighty_mouse/datasets.py`), no `configs/datasets/aind-foraging-bot.yaml`, and no
> `VERSION.txt`/`versions/` yet (`bump_version.py` not run).

## Source format

`_raw/_dlc/aind-forage/Foraging_Bot_UpdatingOld_20220927-LucasKinsey-2022-09-27`: one standard
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
- **Videos**: only those present in the source are copied to `videos/` / `videos_test/`; 7
  labeled videos have no file there; all are in train (their frames are still used).

## Running

```bash
python scripts/preprocess/aind-foraging-bot/convert_aind_foraging_bot.py [--seed N]
```

Spot-checked by overlaying labels on 6 frames (3 train, 3 test).
