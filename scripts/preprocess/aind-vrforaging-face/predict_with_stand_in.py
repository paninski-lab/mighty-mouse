#!/usr/bin/env python3
"""
Run the head-fixed corpus model on _raw/aind-vrforaging-face, posing as one of the model's
training datasets.

The corpus model is dataset-conditioned: it only accepts image paths of the form
`labeled-data/<dataset>/<session>/<frame>` with <dataset> in its `data.dataset_names`, and
this dataset isn't one of them. So the images are exposed through a temp data dir as
`labeled-data/<stand_in>/<session>/...`, predicted, and the predictions saved with the plain
`labeled-data/<session>/<frame>` paths of the raw dataset, ready for

    python scripts/preprocess/transfer_pseudo_labels.py --predictions_csvs <out>... \
        --target_dataset aind-vrforaging-face --keypoints ...

Which dataset to pose as is a judgement call (it picks the dataset embedding); compare
candidates with --csv on a sample before running the full set.

Usage (pose env):
    python scripts/preprocess/aind-vrforaging-face/predict_with_stand_in.py \
        --model_dir results/head-fixed_v9/nonlinear_head --stand_in ibl --out_dir <dir>
"""

import argparse
import tempfile
from pathlib import Path

from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths

DATASET = "aind-vrforaging-face"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--stand_in", required=True, help="a name in the model's dataset_names")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--csvs", nargs="+", default=["CollectedData.csv", "CollectedData_test.csv"]
    )
    parser.add_argument(
        "--n_sample", type=int, default=None, help="only every k-th row, to compare"
    )
    args = parser.parse_args()

    from lightning_pose.api import Model  # slow import; pose env only

    raw = Path(load_paths()["raw_dir"]) / DATASET
    model = Model.from_dir2(args.model_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "labeled-data" / args.stand_in).mkdir(parents=True)
        for session in (raw / "labeled-data").iterdir():
            (tmp / "labeled-data" / args.stand_in / session.name).symlink_to(session)
        for name in args.csvs:
            df = read_labels_csv(raw / name)
            if args.n_sample:
                df = df.iloc[:: args.n_sample]
            df.index = df.index.str.replace("labeled-data/", f"labeled-data/{args.stand_in}/", n=1)
            # distinct basename: predict_on_label_csv keys its output dir off it
            csv = tmp / f"{DATASET}__{args.stand_in}__{name}"
            # the datamodule's sampler needs labeled cells per dataset (it errors on an all-empty
            # CSV); the values are never used for inference
            df.fillna(1.0).to_csv(csv)
            preds = model.predict_on_label_csv(
                csv_file=csv, data_dir=tmp, compute_metrics=False, add_train_val_test_set=False
            ).predictions
            preds.index = preds.index.str.replace(
                f"labeled-data/{args.stand_in}/", "labeled-data/", n=1
            )
            out = args.out_dir / name
            preds.to_csv(out)
            print(f"wrote {out} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
