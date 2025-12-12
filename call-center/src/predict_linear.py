"""Standalone inference for the unified linear pipeline.

Reads test.csv, applies the SAME preprocessing + document building as training,
loads model.joblib, and writes a submission CSV with exactly:
  Id,Prediction

Example:
  python call-center/src/predict_linear.py --run_dir outputs/callcenter/linear/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import linear_pipeline
from config import default_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None, help="Folder with test.csv")
    p.add_argument("--test_csv", type=str, default="test.csv")

    p.add_argument("--run_dir", type=str, required=True, help="Run directory containing model.joblib")

    p.add_argument("--out_path", type=str, default=None, help="Output submission csv path")
    p.add_argument("--id_col", type=str, default="Id")
    p.add_argument("--prediction_col", type=str, default="Prediction")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = default_config()
    data_dir = Path(args.data_dir) if args.data_dir else linear_pipeline.default_local_data_dir()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.joblib: {model_path}")

    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (installed with scikit-learn).") from e

    df_test_raw = pd.read_csv(data_dir / str(args.test_csv))
    df_test_clean = linear_pipeline.preprocess_raw_df(df_test_raw, cfg)

    # Stable ordering by id if present (but we keep original order for submission alignment).
    # We'll output ids from the raw file to avoid dtype surprises.
    if "Id" in df_test_raw.columns:
        out_ids = df_test_raw["Id"]
    elif "id" in df_test_raw.columns:
        out_ids = df_test_raw["id"]
    elif cfg.id_col in df_test_clean.columns:
        out_ids = df_test_clean[cfg.id_col]
    else:
        raise ValueError("Could not find an id column in test.csv")

    model = joblib.load(model_path)
    pred = model.predict(df_test_clean)
    pred = np.asarray(pred).reshape(-1).astype(int)

    out = pd.DataFrame({str(args.id_col): out_ids, str(args.prediction_col): pred})

    out_path = Path(args.out_path) if args.out_path else (run_dir / "submission.csv")
    out.to_csv(out_path, index=False)
    print(f"[predict_linear] wrote: {out_path}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
