# hantman-mv dataset changelog

See [`README.md`](README.md) for the conversion pipeline and design decisions.

## Changelog

### 2026-10-17 (MW) (version 2)
- Added `d1_tip_left`, `d2_tip_left`, `d3_tip_left`, `d4_tip_left` and `wrist_left` to
  `project.yaml` + both `CollectedData*.csv` files. Fingertips come from SG's
  pseudo-labeling plus manual adjustment (front view only; side view not labeled or
  reviewed), and were further edited manually afterward. `wrist_left` was labeled manually
  on both views.
- Labeled counts (front view, of 96 train / 15 test frames): `d1`–`d4_tip_left` 51–59 train,
  6–9 test; `wrist_left` 75 train, 9 test. Side view: no fingertips, `wrist_left` in 51
  train and 2 test frames.
- No existing column changed (0 cells differ from version 1), including `nose_top`; SG's
  separate `nose_tip`/`hand_*`/`wrist` edits in `tmp/` were not taken.
- Pipeline changes to support the new left-paw labels:
  - [`configs/datasets/hantman-mv.yaml`](../../../configs/datasets/hantman-mv.yaml): mapped
    `d{1-4}_tip_left` and `wrist_left` explicitly, and switched `d{1-4}_tip` / `wrist_new`
    from `*_{side}` to plain `*_right` targets. A `{side}` template also emits an all-NaN
    `*_left` column, which would collide with the real one (all sessions are `right`, so
    the right paw is unaffected).
  - `_post_process_hantman_mv` (`mighty_mouse/convert.py`) is now view-aware:
    `d{1-4}_tip_left` get `visible=0` on `*_side` frames (never labeled or reviewed) and keep
    the default `visible=1` on `*_front` frames (every frame reviewed, so empty = occluded);
    `wrist_left` is exempt on both views; all other `_left` columns (e.g. `eye_*_left`)
    are still forced to `visible=0` except `ear_*_left`.
  - New `check_duplicate_targets` validation: two source keypoints may not resolve to the
    same canonical name.

### 2026-09-29 (MW) (version 1)
- Added `nose_top` as a new keypoint to `project.yaml` + both `CollectedData*.csv` files
  (via `scripts/add_keypoints.py`), and manually labeled it in the LP app in every frame
  (192/192 train, 30/30 test, front and side views). The canonical name already existed
  in `configs/keypoints.yaml`/`model.yaml`; mapped `nose_top: nose_top` (midline) in
  [`configs/datasets/hantman-mv.yaml`](../../../configs/datasets/hantman-mv.yaml).
- Manually adjusted `nose_tip` in front-view frames (27 train, 12 test; median shift
  ~4 px, max ~8 px). No side-view `nose_tip` labels changed.

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of hantman-mv's label CSVs. This predates the
  versioning scheme, so v0 is the best-available current state rather than a
  pristine collaborator copy — it already includes the undocumented
  pre-versioning edits logged above (2026-09-15 new eye/nose keypoints +
  manual labels, derived-then-manually-corrected `wrist_new` column;
  2026-09-17 empty `ear_*` keypoints added). Neither of those additions is
  reproduced by `convert_hantman_mv.py` — see the note at the bottom of this
  changelog.

### 2026-09-22 (MW)
- Ran `scripts/preprocessing/extract_clips.py` (see
  [`cazettes-side/CHANGELOG.md`](../cazettes-side/CHANGELOG.md) for where that script came
  from) on the four videos in `_raw/_dlc/hantman-mv/videos_test/`
  (`KPC190_20260214_v033_front`, `KPC190_20260214_v033_side`,
  `KPC200_20260323_v053_front`, `KPC200_20260323_v053_side`), saving to
  `_raw/hantman-mv/videos_test/` as labeling-candidate review clips.
- These containers report `1 fps` (5998 frames -> ffprobe reads the duration as ~100
  minutes), but the true capture rate is 500 fps (~12s actual duration) — the 1 fps tag
  is wrong metadata, not a real frame rate. This prompted two new options on
  `extract_clips.py`/`make_video_snippet`: `--from-start` (take the first
  `--clip-length` seconds instead of searching for the highest-motion window) and
  `--fps` (override the frame rate used for all time math, and pass `-r <fps>` to
  ffmpeg as an input option so it regenerates timestamps at the true rate instead of
  trusting the container's declared one).
- First tried `--clip-length 1 --skip-start 0 --from-start --fps 500` as a quick check
  (500 fps / 500 frames / 1.0s output, confirmed correct), then `--clip-length 12` for
  the full ~12s clip, before settling on `--clip-length 8` — verified via ffprobe that
  each final output clip is 500 fps / 4000 frames / 8.0s duration, h264/yuv420p/mp4.
  Raw-video review only, not a label change — `CollectedData*.csv` files are
  untouched.

### 2026-09-17 (MW)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints (all rows
  empty — ears are never visible in this view) to `project.yaml` + both
  `CollectedData*.csv` files, and mapped `ear_*: ear_*_{side}` in
  [`configs/datasets/hantman-mv.yaml`](../../../configs/datasets/hantman-mv.yaml). All
  four canonical names already existed in `configs/keypoints.yaml`/`model.yaml`.
- Added an exemption (`_HANTMAN_MV_LEFT_SUPPRESS_EXEMPT`) to `_post_process_hantman_mv`
  in `scripts/convert_dataset.py` so `ear_*_left` stays at the default `visible=1`
  (matching `ear_*_right`) instead of being swept into the blanket `_left → visible=0`
  rule described under "Stage 2" in `README.md`. Without it the ear keypoints would've
  ended up asymmetric (right actively suppressed, left excluded from loss entirely),
  unlike the symmetric treatment in `cazettes-side`/`facemap`. The pre-existing
  `eye_*_left`/digit/wrist `_left` columns are unaffected and still forced to
  `visible=0`.

### 2026-09-15 (MW)
- Added `eye_back`, `eye_top`, `eye_front`, `eye_bottom`, `nose_tip`, and `nose_bottom`
  as new keypoints to `project.yaml` + both `CollectedData*.csv` files, and manually
  labeled them in the LP app (no source ever provided values for them).
- Computed a new `wrist_new` column from three existing keypoints:
  `mid = nanmean(d2_base, d3_base)` (x/y independently), then
  `wrist_new = (mid + hand_middle) / 2` — `NaN` whenever either input is `NaN`.
  183/192 train rows and 24/30 test rows got a real value under this formula; some
  values were **manually corrected afterward**, so the CSVs no longer match the formula
  exactly row-for-row — treat it as how the column originated, not a reproducible
  derivation of its current values. Maps to `wrist_{side}` (reuses existing canonical
  `wrist_left`/`wrist_right` — no new vocab needed).

**Neither addition above is reproduced by `convert_hantman_mv.py`.** Both are manual
edits to the stage-1 output; the raw source (`_raw/_dlc/hantman-mv`) still only has the
original 17 finger/paw/pellet keypoints, and a from-scratch re-run would regenerate only
those and silently drop these additions.
