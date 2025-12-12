"""Optuna hyperparameter optimization for the unified linear pipeline.

Produces a fully reproducible run directory:
  outputs/callcenter/linear/<run_id>/
    model.joblib
    best_params.json
    metrics.json
    feature_spec.json
    train_config.json

Metric: Macro-F1 (everywhere).

Example:
  python call-center/src/optuna_search.py --trials 50
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import linear_pipeline
from config import default_config


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--team_dir", type=str, default=None, help="Drive root (Colab), e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--data_dir", type=str, default=None, help="Folder with train.csv/test.csv/sample_submission.csv")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--test_csv", type=str, default="test.csv")

    p.add_argument("--out_dir", type=str, default=None, help="Base outputs dir (default: outputs/callcenter/linear)")
    p.add_argument("--run_id", type=str, default=None)

    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)

    # document toggles
    p.add_argument("--add_structured_tokens", action="store_true", help="Enable COL=VALUE tokens")
    p.add_argument("--add_time_tokens", action="store_true", help="Enable HOUR/DOW/MONTH tokens")

    # optuna
    p.add_argument("--study_name", type=str, default="callcenter_linear")
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])

    return p.parse_args()


def _set_repro(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    np.random.seed(int(seed))


def _load_train(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = default_config()
    df_raw = pd.read_csv(data_dir / "train.csv")
    df_clean = linear_pipeline.preprocess_raw_df(df_raw, cfg)
    if cfg.target_col not in df_clean.columns:
        raise ValueError(f"Missing target column in cleaned train: {cfg.target_col}")
    y = pd.to_numeric(df_clean[cfg.target_col], errors="raise").astype(int).to_numpy()
    # Ensure deterministic ordering (by id if present)
    if cfg.id_col in df_clean.columns:
        df_clean = df_clean.sort_values(cfg.id_col, kind="mergesort").reset_index(drop=True)
        y = pd.to_numeric(df_clean[cfg.target_col], errors="raise").astype(int).to_numpy()
    return df_clean, y


def main() -> None:
    args = parse_args()
    _set_repro(int(args.seed))

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = linear_pipeline.resolve_data_dir(team_dir=args.team_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    base_out = Path(args.out_dir) if args.out_dir else linear_pipeline.resolve_outputs_dir(team_dir=args.team_dir)
    run_id = args.run_id or _utc_run_id()
    run_dir = base_out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional dependency
    try:
        import optuna  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Optuna is not installed. Install with: pip install optuna\n"
            f"Original error: {e}"
        ) from e

    cfg = default_config()
    doc_cfg = linear_pipeline.DocumentConfig(
        add_structured_tokens=bool(args.add_structured_tokens),
        add_time_tokens=bool(args.add_time_tokens),
    )

    df_clean, y = _load_train(data_dir)

    labels = sorted(np.unique(y).tolist())

    def objective(trial: "optuna.Trial") -> float:
        # ----- Search space -----
        word_ngram = trial.suggest_categorical("word_ngram", [(1, 1), (1, 2)])
        char_ngram = trial.suggest_categorical("char_ngram", [(3, 5), (4, 6)])

        min_df = trial.suggest_int("min_df", 1, 8)
        max_df = trial.suggest_float("max_df", 0.85, 0.98)

        # word/char caps (log-ish)
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

        pipe = linear_pipeline.build_unified_pipeline(
            cfg=cfg,
            doc_cfg=doc_cfg,
            word_ngram_range=word_ngram,
            char_ngram_range=char_ngram,
            min_df=min_df,
            max_df=max_df,
            max_features_word=max_features_word,
            max_features_char=max_features_char,
            sublinear_tf=sublinear_tf,
            classifier=clf_choice,
            clf_params=clf_params,
            seed=int(args.seed),
        )

        cv = linear_pipeline.cv_score_macro_f1(
            df_clean=df_clean,
            y=y,
            pipeline=pipe,
            n_splits=int(args.folds),
            seed=int(args.seed),
            class_balance=class_balance,
        )
        # Optimize by OOF macro-f1 (more stable than fold mean)
        trial.set_user_attr("cv", cv)
        return float(cv["oof_macro_f1"])

    # Sampler
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

    pipe = linear_pipeline.build_unified_pipeline(
        cfg=cfg,
        doc_cfg=doc_cfg,
        word_ngram_range=tuple(best_params["word_ngram"]),
        char_ngram_range=tuple(best_params["char_ngram"]),
        min_df=int(best_params["min_df"]),
        max_df=float(best_params["max_df"]),
        max_features_word=int(best_params["max_features_word"]),
        max_features_char=int(best_params["max_features_char"]),
        sublinear_tf=bool(best_params["sublinear_tf"]),
        classifier=clf_choice,
        clf_params=clf_params,
        seed=int(args.seed),
    )

    sw = linear_pipeline.compute_sample_weight(y, mode=class_balance)
    fit_params = {"clf__sample_weight": sw} if sw is not None else {}
    pipe.fit(df_clean, y, **fit_params)

    # Save artifacts
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (installed with scikit-learn).") from e

    model_path = run_dir / "model.joblib"
    joblib.dump(pipe, model_path, compress=3)

    linear_pipeline.write_json(run_dir / "best_params.json", {
        "best_value_oof_macro_f1": float(best.value),
        "params": best_params,
        "resolved": {
            "classifier": clf_choice,
            "class_balance": class_balance,
            "clf_params": clf_params,
        },
    })

    linear_pipeline.write_json(run_dir / "metrics.json", {
        "metric": "macro_f1",
        "oof_macro_f1": float(best_cv.get("oof_macro_f1", best.value)),
        "macro_f1_mean": float(best_cv.get("macro_f1_mean", 0.0)),
        "macro_f1_std": float(best_cv.get("macro_f1_std", 0.0)),
        "per_class_f1": best_cv.get("per_class_f1"),
        "labels": labels,
        "folds": best_cv.get("folds"),
    })

    linear_pipeline.write_json(run_dir / "feature_spec.json", {
        "text_col": cfg.text_col,
        "datetime_col": cfg.datetime_col,
        "structured_cols": list(doc_cfg.structured_cols),
        "add_structured_tokens": bool(doc_cfg.add_structured_tokens),
        "add_time_tokens": bool(doc_cfg.add_time_tokens),
        "add_text": bool(doc_cfg.add_text),
        "notes": {
            "sn_not_used": True,
            "id_not_used": True,
        },
    })

    linear_pipeline.write_json(run_dir / "train_config.json", {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "folds": int(args.folds),
        "data_dir": str(data_dir),
        "doc_cfg": asdict(doc_cfg),
    })

    print(f"[optuna_search] Saved run_dir: {run_dir}")
    print(f"[optuna_search] Best OOF Macro-F1: {float(best.value):.6f}")


if __name__ == "__main__":
    main()
