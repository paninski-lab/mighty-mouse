# aind-vrforaging-face

> **Status: stage 1 only.** `_raw/aind-vrforaging-face/` is a usable standalone LP project, not yet in
> the combined corpus. See [`skills/preprocess/preprocess-new-dataset/SKILL.md`](../../../skills/preprocess/preprocess-new-dataset/SKILL.md) for what
> stage 2 would involve; don't start it unless asked.

Face-camera frames from head-fixed mice in the AIND VR foraging task, with pseudo-labels from the
head-fixed corpus model. See [`CHANGELOG.md`](CHANGELOG.md) for label history.

## Source

`_raw/_dlc/aind-vrforaging/face-camera/` is a DeepLabCut project (`face_camera_patch_foraging`) with
619 frames in `labeled-data/<session>/`, a three-quarter view of the face with the mouse's
head-fixed implant and the spout. Only one session (20 frames) was ever labeled, so those labels
are discarded and only the frames are used.

## Pipeline

1. `create_aind_vrforaging_face_dataset.py` copies the frames (native sizes, 720x540 and
   1440x1080 mixed), writes empty `CollectedData*.csv` with the `KEYPOINTS` columns (canonical
   names, so the dataset config maps them one-to-one) and splits by mouse.
   - The two `<session> (1)` folders hold different frames of the same sessions at another crop
     (1440x1080), and are merged into `<session>`.
   - The mouse id is in the session name (`behavior_<mouse>_<date>`), so the split holds out whole
     mice. The `video` and un-dated `behavior_754575` folders have no session identity; `video` is
     its own subject, and `behavior_754575` goes with mouse 754575.
2. `predict_with_stand_in.py` runs `results/head-fixed_v11/mightymouse-v11-384` on every frame. That
   model only accepts paths whose dataset is in its `dataset_names`, so the images are presented as
   another dataset's; the choice doesn't matter (different stand-ins gave byte-identical
   predictions).
3. `transfer_pseudo_labels.py --predictions_csvs ...` fills the keypoints below where likelihood
   >= 0.7.

Keypoints (all right-side, plus both wrists): `eye_{back,bottom,front,top}_right`, `nose_{tip,top,bottom}`,
`pad_{top,side}_right`, `ear_{top,tip,bottom,base}_right`, `wrist_{left,right}`.
