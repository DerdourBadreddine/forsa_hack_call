"""Optuna HPO for the strong LinearSVC-style sparse pipeline.

This matches the original baseline structure from `train_linear_svc.py`:
- TF-IDF char_wb + TF-IDF word on cleaned text
- OneHotEncoder on categorical columns
- Engineered numeric features (time + text helpers)

But it wraps feature engineering inside the pipeline so the exported
`model.joblib` is self-contained for inference.

Outputs (required):
  outputs/callcenter/linear/<run_id>/
    model.joblib
    best_params.json
    metrics.json
    feature_spec.json
    train_config.json

Metric: Macro-F1 via StratifiedKFold.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import features
import linear_pipeline
import linear_svc_pipeline
from config import default_config


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _parse_pair(s: str) -> tuple[int, int]:
    a, b = [x.strip() for x in str(s).split(",", 1)]
    return int(a), int(b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default=None, help="Folder with train.csv/test.csv")
    p.add_argument("--train_csv", type=str, default="train.csv")

    p.add_argument("--out_dir", type=str, default=None, help="Base outputs dir (default: outputs/callcenter/linear)")
    p.add_argument("--run_id", type=str, default=None)

    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)

    # optuna
    p.add_argument("--study_name", type=str, default="callcenter_linear_svc")
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])

    return p.parse_args()


def _set_repro(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    np.random.seed(int(seed))


def _load_train(data_dir: Path, train_csv: str) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = default_config()
    df_raw = pd.read_csv(data_dir / str(train_csv))
    df_clean = linear_pipeline.preprocess_raw_df(df_raw, cfg)
    if cfg.id_col in df_clean.columns:
        df_clean = df_clean.sort_values(cfg.id_col, kind="mergesort").reset_index(drop=True)

    if cfg.target_col not in df_clean.columns:
        raise ValueError(f"Missing target column in cleaned train: {cfg.target_col}")
    y = pd.to_numeric(df_clean[cfg.target_col], errors="raise").astype(int).to_numpy()
    return df_clean, y


def main() -> None:
    args = parse_args()
    _set_repro(int(args.seed))

    data_dir = Path(args.data_dir) if args.data_dir else linear_pipeline.default_local_data_dir()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    base_out = Path(args.out_dir) if args.out_dir else linear_pipeline.default_outputs_dir()
    run_id = args.run_id or _utc_run_id()
    run_dir = base_out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional dependency
    try:
        import optuna  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Optuna is not installed. Install with: pip install optuna") from e

    cfg = default_config()
    base_spec = features.build_feature_spec(cfg)
    spec = linear_svc_pipeline.LinearSvcFeatureSpec(
        text_col=str(base_spec["text_col"]),
        cat_cols=tuple(base_spec["cat_cols"]),
        num_cols=tuple(base_spec["num_cols"]),
    )

    df_clean, y = _load_train(data_dir, args.train_csv)
    labels = sorted(np.unique(y).tolist())

    def objective(trial: "optuna.Trial") -> float:
        word_ngram = trial.suggest_categorical("word_ngram", ["1,1", "1,2"])
        char_ngram = trial.suggest_categorical("char_ngram", ["3,5", "4,6"])

        min_df = trial.suggest_int("min_df", 1, 8)
        max_df = trial.suggest_float("max_df", 0.85, 0.98)

        max_features_word = int(trial.suggest_int("max_features_word", 80_000, 200_000, step=10_000))
        max_features_char = int(trial.suggest_int("max_features_char", 150_000, 300_000, step=10_000))

        sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])

        clf_choice = trial.suggest_categorical("classifier", ["logreg", "linearsvc", "sgd"])
        class_balance = trial.suggest_categorical("class_balance", ["none", "balanced", "custom_sqrt"])

        clf_params: dict[str, Any] = {}
        if clf_choice == "logreg":
            clf_params["C"] = trial.suggest_float("logreg_C", 0.2, 12.0, log=True)
            clf_params["penalty"] = trial.suggest_categorical("logreg_penalty", ["l2", "elasticnet"])
            if clf_params["penalty"] == "elasticnet":
                clf_params["l1_ratio"] = trial.suggest_float("logreg_l1_ratio", 0.05, 0.5)
            clf_params["max_iter"] = 4000
            clf_params["n_jobs"] = -1
        elif clf_choice == "linearsvc":
            clf_params["C"] = trial.suggest_float("svm_C", 0.2, 12.0, log=True)
            clf_params["max_iter"] = 12000
        else:
            clf_params["alpha"] = trial.suggest_float("sgd_alpha", 1e-6, 1e-4, log=True)
            clf_params["penalty"] = trial.suggest_categorical("sgd_penalty", ["l2", "elasticnet"])
            if clf_params["penalty"] == "elasticnet":
                clf_params["l1_ratio"] = trial.suggest_float("sgd_l1_ratio", 0.05, 0.3)
            clf_params["loss"] = trial.suggest_categorical("sgd_loss", ["log_loss", "hinge"])
            clf_params["max_iter"] = 2000
            clf_params["tol"] = 1e-3

        pipe = linear_svc_pipeline.build_pipeline(
            cfg=cfg,
            spec=spec,
            word_ngram_range=_parse_pair(word_ngram),
            char_ngram_range=_parse_pair(char_ngram),
            min_df=min_df,
            max_df=max_df,
            max_features_word=max_features_word,
            max_features_char=max_features_char,
            sublinear_tf=sublinear_tf,
            classifier=clf_choice,
            clf_params=clf_params,
            seed=int(args.seed),
        )

        cv = linear_svc_pipeline.cv_score_macro_f1(
            cfg=cfg,
            df_clean=df_clean,
            y=y,
            pipeline=pipe,
            n_splits=int(args.folds),
            seed=int(args.seed),
            class_balance=class_balance,
        )
        trial.set_user_attr("cv", cv)
        return float(cv["oof_macro_f1"])

    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=int(args.seed))
    else:
        sampler = optuna.samplers.TPESampler(seed=int(args.seed))

    study = optuna.create_study(direction="maximize", study_name=str(args.study_name), sampler=sampler)
    study.optimize(objective, n_trials=int(args.trials), timeout=int(args.timeout) if args.timeout else None)

    best = study.best_trial
    best_params = dict(best.params)
    best_cv = dict(best.user_attrs.get("cv") or {})

    # Rebuild best pipeline
    clf_choice = best_params["classifier"]
    class_balance = best_params["class_balance"]

    clf_params: dict[str, Any] = {}
    if clf_choice == "logreg":
        clf_params = {
            "C": float(best_params["logreg_C"]),
            "penalty": str(best_params["logreg_penalty"]),
            "max_iter": 4000,
            "n_jobs": -1,
        }
        if clf_params["penalty"] == "elasticnet":
            clf_params["l1_ratio"] = float(best_params["logreg_l1_ratio"])
    elif clf_choice == "linearsvc":
        clf_params = {"C": float(best_params["svm_C"]), "max_iter": 12000}
    else:
        clf_params = {
            "alpha": float(best_params["sgd_alpha"]),
            "penalty": str(best_params["sgd_penalty"]),
            "loss": str(best_params["sgd_loss"]),
            "max_iter": 2000,
            "tol": 1e-3,
        }
        if clf_params["penalty"] == "elasticnet":
            clf_params["l1_ratio"] = float(best_params["sgd_l1_ratio"])

    pipe = linear_svc_pipeline.build_pipeline(
        cfg=cfg,
        spec=spec,
        word_ngram_range=_parse_pair(best_params["word_ngram"]),
        char_ngram_range=_parse_pair(best_params["char_ngram"]),
        min_df=int(best_params["min_df"]),
        max_df=float(best_params["max_df"]),
        max_features_word=int(best_params["max_features_word"]),
        max_features_char=int(best_params["max_features_char"]),
        sublinear_tf=bool(best_params["sublinear_tf"]),
        classifier=clf_choice,
        clf_params=clf_params,
        seed=int(args.seed),
    )

    sw = linear_svc_pipeline.compute_sample_weight(y, mode=class_balance)
    fit_params = {"clf__sample_weight": sw} if sw is not None else {}
    pipe.fit(df_clean, y, **fit_params)

    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (installed with scikit-learn).") from e

    joblib.dump(pipe, run_dir / "model.joblib", compress=3)

    linear_pipeline.write_json(
        run_dir / "best_params.json",
        {
            "best_value_oof_macro_f1": float(best.value),
            "params": best_params,
            "resolved": {
                "classifier": clf_choice,
                "class_balance": class_balance,
                "clf_params": clf_params,
            },
        },
    )

    linear_pipeline.write_json(
        run_dir / "metrics.json",
        {
            "metric": "macro_f1",
            "oof_macro_f1": float(best_cv.get("oof_macro_f1", best.value)),
            "macro_f1_mean": float(best_cv.get("macro_f1_mean", 0.0)),
            "macro_f1_std": float(best_cv.get("macro_f1_std", 0.0)),
            "per_class_f1": best_cv.get("per_class_f1"),
            "labels": labels,
            "folds": best_cv.get("folds"),
            "sanity": best_cv.get("sanity"),
        },
    )

    linear_pipeline.write_json(
        run_dir / "feature_spec.json",
        {
            "pipeline_type": "linear_svc_style",
            "text_col": spec.text_col,
            "cat_cols": list(spec.cat_cols),
            "num_cols": list(spec.num_cols),
            "notes": {
                "id_not_used": True,
                "sn_not_used": True,
            },
        },
    )

    linear_pipeline.write_json(
        run_dir / "train_config.json",
        {
            "run_id": run_id,
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
            "folds": int(args.folds),
            "data_dir": str(data_dir),
            "cfg": asdict(cfg),
        },
    )

    print(f"[optuna_search_linear_svc] Saved run_dir: {run_dir}")
    print(f"[optuna_search_linear_svc] Best OOF Macro-F1: {float(best.value):.6f}")


if __name__ == "__main__":
    main()
