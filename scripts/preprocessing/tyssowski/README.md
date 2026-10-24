# tyssowski preprocessing

Builds `_raw/tyssowski` (single-view) from the multi-view DLC project at
`_raw/_dlc/tyssowski`.

**Status: in the corpus (stage 2 done), not built.** `configs/datasets/tyssowski.yaml`
exists, `"tyssowski"` is in `ALL_DATASETS`, and `scripts/convert_dataset.py --dataset tyssowski`
has been run; `scripts/build_dataset.py` has not.

## Why a custom converter

Another variant of the `hantman-mv` reaching task (side + front views, per-view CSVs,
`view_names: [front, side]`), so the conversion is the same as
[`../hantman-mv/`](../hantman-mv/README.md): concat the two CSVs (view is already encoded in
the session names) and write a `project.yaml` with `view_names: []`. Differences:

- **Different skeleton**: `d1` (thumb), `d{2-5}_{1,2,3}` (`_3` = tip), `hand`, `wrist`,
  `elbow`, `shoulder`, `pellet`, `nose`, `mouth`.
- **Unused keypoints are dropped from the output CSVs and `project.yaml`** (not listed in an
  `exclude` list) to keep the labeling app uncluttered. Only `KEEP_KEYPOINTS` in the script
  survive: `d{2-5}_3`, `hand`, `nose`, `mouth`.
- Source names are kept; mapping is in `configs/datasets/tyssowski.yaml`:
  `d2_3`→`d1_tip_right`, `d3_3`→`d2_tip_right`, `d4_3`→`d3_tip_right`,
  `d5_3`→`d4_tip_right` (the canonical `d1_tip` is the first finger, so the source
  thumb has no canonical home), `hand`→`wrist_right`, `nose`→`nose_tip`. `mouth` stays in the raw CSVs for labeling but is in
  `exclude.keypoints`, so it isn't in the corpus.
  Laterality (`right`, all sessions) was confirmed by the user, not inferred.

## Design notes

- **Split:** subject-level (first `_` token, e.g. `M399`), target 0.125. Seed 0 gave 15.7%
  test (just out of the 10-15% range), so the default is **seed 1** (30/204 = 14.7%).
- **Left paw**: not labeled yet, so `POST_PROCESS["tyssowski"]` (`mighty_mouse/convert.py`)
  forces `wrist_left`/`d{1-4}_tip_left` to `visible=0` (annotation gap, not occlusion).
  Once left fingertips are labeled, revisit (see `hantman-mv` for the by-view handling).
- `project.yaml` carries a `skeleton` connecting `hand` (wrist) to each digit tip.
- Only CSV-referenced images are copied; `videos/` is created empty.

See [`CHANGELOG.md`](CHANGELOG.md) for label history.

## Usage

```bash
python scripts/preprocessing/tyssowski/convert_tyssowski.py [--seed N]
python scripts/convert_dataset.py --dataset tyssowski
```
