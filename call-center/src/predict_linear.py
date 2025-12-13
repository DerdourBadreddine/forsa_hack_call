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

from . import linear_pipeline
from .config import default_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None, help="Drive root (Colab), e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--data_dir", type=str, default=None, help="Folder with test.csv")
    p.add_argument("--test_csv", type=str, default="test.csv")

    p.add_argument("--sample_submission_csv", type=str, default="sample_submission.csv", help="Used to enforce Id ordering")

    p.add_argument("--run_dir", type=str, required=True, help="Run directory containing model.joblib")

    p.add_argument("--out_path", type=str, default=None, help="Output submission csv path")
    p.add_argument("--id_col", type=str, default="Id")
    p.add_argument("--prediction_col", type=str, default="Prediction")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = default_config()
    data_dir = Path(args.data_dir) if args.data_dir else linear_pipeline.resolve_data_dir(team_dir=args.team_dir)
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

    test_path = data_dir / str(args.test_csv)
    df_test_raw = pd.read_csv(test_path)
    df_test_clean = linear_pipeline.preprocess_raw_df(df_test_raw, cfg)

    # Use sample_submission to enforce exact ordering when available.
    sample_path = data_dir / str(args.sample_submission_csv)
    df_sample = pd.read_csv(sample_path) if sample_path.exists() else None

    # Determine test ids
    if "Id" in df_test_raw.columns:
        test_ids = df_test_raw["Id"]
    elif "id" in df_test_raw.columns:
        test_ids = df_test_raw["id"]
    elif cfg.id_col in df_test_clean.columns:
        test_ids = df_test_clean[cfg.id_col]
    else:
        raise ValueError(f"Could not find an id column in {test_path}")

    model = joblib.load(model_path)
    pred = model.predict(df_test_clean)
    pred = np.asarray(pred).reshape(-1).astype(int)

    pred_df = pd.DataFrame({"Id": test_ids.astype("string"), "Prediction": pred})
    if df_sample is not None:
        # Accept common variants but enforce final output columns.
        if df_sample.columns.tolist() == ["Id", "Prediction"]:
            out = df_sample[["Id"]].merge(pred_df, on="Id", how="left")
        elif df_sample.columns.tolist() == ["id", "class_int"]:
            tmp = df_sample.rename(columns={"id": "Id"})[["Id"]]
            out = tmp.merge(pred_df, on="Id", how="left")
        else:
            tmp = df_sample[[df_sample.columns[0]]].rename(columns={df_sample.columns[0]: "Id"})
            out = tmp.merge(pred_df, on="Id", how="left")

        if out["Prediction"].isna().any():
            missing = int(out["Prediction"].isna().sum())
            raise RuntimeError(f"Failed to align predictions to sample_submission ids; missing={missing}")
        out["Prediction"] = out["Prediction"].astype(int)
    else:
        out = pred_df.copy()

    # Final column names required by the hackathon: Id,Prediction
    out = out.rename(columns={"Id": str(args.id_col), "Prediction": str(args.prediction_col)})

    out_path = Path(args.out_path) if args.out_path else (run_dir / "submission.csv")
    out.to_csv(out_path, index=False)
    print(f"[predict_linear] wrote: {out_path}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
