"""Train the unified linear pipeline with explicit hyperparameters (no Optuna).

Use this after you have best_params.json, or for quick baselines.

Outputs to:
  outputs/callcenter/linear/<run_id>/

Metric: Macro-F1 (printed as OOF via CV).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import linear_pipeline
from config import default_config


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--team_dir", type=str, default=None, help="Drive root (Colab), e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)

    p.add_argument("--add_structured_tokens", action="store_true")
    p.add_argument("--add_time_tokens", action="store_true")

    p.add_argument("--word_ngram", type=str, default="1,2")
    p.add_argument("--char_ngram", type=str, default="3,5")
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    p.add_argument("--max_features_word", type=int, default=120000)
    p.add_argument("--max_features_char", type=int, default=250000)
    p.add_argument("--sublinear_tf", action="store_true")

    p.add_argument("--classifier", type=str, default="linearsvc", choices=["linearsvc", "logreg", "sgd"])
    p.add_argument("--class_balance", type=str, default="balanced", choices=["none", "balanced", "custom_sqrt"])

    # classifier params
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--penalty", type=str, default="l2")
    p.add_argument("--l1_ratio", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1e-5)
    p.add_argument("--loss", type=str, default="log_loss")

    return p.parse_args()


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = [x.strip() for x in str(s).split(",", 1)]
    return int(a), int(b)


def main() -> None:
    args = parse_args()
    cfg = default_config()

    data_dir = Path(args.data_dir) if args.data_dir else linear_pipeline.resolve_data_dir(team_dir=args.team_dir)
    out_dir = Path(args.out_dir) if args.out_dir else linear_pipeline.resolve_outputs_dir(team_dir=args.team_dir)
    run_id = args.run_id or _utc_run_id()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(data_dir / "train.csv")
    df_clean = linear_pipeline.preprocess_raw_df(df_raw, cfg)
    if cfg.id_col in df_clean.columns:
        df_clean = df_clean.sort_values(cfg.id_col, kind="mergesort").reset_index(drop=True)

    y = pd.to_numeric(df_clean[cfg.target_col], errors="raise").astype(int).to_numpy()

    doc_cfg = linear_pipeline.DocumentConfig(
        add_structured_tokens=bool(args.add_structured_tokens),
        add_time_tokens=bool(args.add_time_tokens),
    )

    clf_params = {}
    if args.classifier == "linearsvc":
        clf_params = {"C": float(args.C), "max_iter": 12000}
    elif args.classifier == "logreg":
        clf_params = {
            "C": float(args.C),
            "penalty": str(args.penalty),
            "l1_ratio": float(args.l1_ratio),
            "max_iter": 4000,
            "n_jobs": -1,
        }
    else:
        clf_params = {
            "alpha": float(args.alpha),
            "penalty": str(args.penalty),
            "l1_ratio": float(args.l1_ratio),
            "loss": str(args.loss),
            "max_iter": 2000,
            "tol": 1e-3,
        }

    pipe = linear_pipeline.build_unified_pipeline(
        cfg=cfg,
        doc_cfg=doc_cfg,
        word_ngram_range=_parse_pair(args.word_ngram),
        char_ngram_range=_parse_pair(args.char_ngram),
        min_df=int(args.min_df),
        max_df=float(args.max_df),
        max_features_word=int(args.max_features_word),
        max_features_char=int(args.max_features_char),
        sublinear_tf=bool(args.sublinear_tf),
        classifier=str(args.classifier),
        clf_params=clf_params,
        seed=int(args.seed),
    )

    cv = linear_pipeline.cv_score_macro_f1(
        df_clean=df_clean,
        y=y,
        pipeline=pipe,
        n_splits=int(args.folds),
        seed=int(args.seed),
        class_balance=str(args.class_balance),
    )

    # Fit on full data
    sw = linear_pipeline.compute_sample_weight(y, mode=str(args.class_balance))
    fit_params = {"clf__sample_weight": sw} if sw is not None else {}
    pipe.fit(df_clean, y, **fit_params)

    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (installed with scikit-learn).") from e

    joblib.dump(pipe, run_dir / "model.joblib", compress=3)
    linear_pipeline.write_json(run_dir / "metrics.json", cv)
    linear_pipeline.write_json(run_dir / "feature_spec.json", {
        "text_col": cfg.text_col,
        "datetime_col": cfg.datetime_col,
        "structured_cols": list(doc_cfg.structured_cols),
        "add_structured_tokens": bool(doc_cfg.add_structured_tokens),
        "add_time_tokens": bool(doc_cfg.add_time_tokens),
        "add_text": bool(doc_cfg.add_text),
    })

    print(f"[train_linear] run_dir={run_dir}")
    print(f"[train_linear] OOF Macro-F1={float(cv['oof_macro_f1']):.6f}  mean={float(cv['macro_f1_mean']):.6f}±{float(cv['macro_f1_std']):.6f}")


if __name__ == "__main__":
    main()
