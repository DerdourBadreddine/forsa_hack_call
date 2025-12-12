"""LinearSVC-style sparse pipeline (text TF-IDF + OHE cats + engineered numeric).

This keeps the strong original baseline structure from `train_linear_svc.py`, but
wraps feature engineering inside a pickle-safe sklearn Pipeline so the saved
`model.joblib` is truly standalone for inference.

Input to the pipeline: a cleaned canonical DataFrame (output of preprocess).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

import features
from config import CallCenterConfig


@dataclass(frozen=True)
class LinearSvcFeatureSpec:
    text_col: str
    cat_cols: tuple[str, ...]
    num_cols: tuple[str, ...]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer numeric features and enforce stable dtypes/columns."""

    def __init__(self, cfg: CallCenterConfig, spec: LinearSvcFeatureSpec):
        self.cfg = cfg
        self.spec = spec

    def fit(self, X: pd.DataFrame, y: Any = None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df_feat = features.engineer_features(X.copy(), self.cfg)

        text_col = str(self.spec.text_col)
        cat_cols = list(self.spec.cat_cols)
        num_cols = list(self.spec.num_cols)

        df_feat[text_col] = df_feat[text_col].astype("string").fillna("")
        for c in cat_cols:
            df_feat[c] = df_feat[c].astype("string").fillna("UNK")
        for c in num_cols:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").fillna(0.0).astype("float32")

        cols = [text_col, *cat_cols, *num_cols]
        missing = [c for c in cols if c not in df_feat.columns]
        if missing:
            raise ValueError(f"Missing engineered feature columns: {missing}")
        return df_feat[cols].copy()


def compute_sample_weight(y: np.ndarray, *, mode: str) -> np.ndarray | None:
    """Compute sample_weight for imbalance handling.

    mode:
      - 'none'
      - 'balanced' (sklearn balanced)
      - 'custom_sqrt' (w_c = 1/sqrt(freq_c), normalized to mean 1)
    """

    mode = str(mode).lower().strip()
    if mode in {"none", "null", ""}:
        return None

    y = np.asarray(y).astype(int)
    classes = np.unique(y)

    if mode == "balanced":
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        w_by_class = {int(c): float(w) for c, w in zip(classes, cw)}
    elif mode in {"custom_sqrt", "sqrt"}:
        freq = {int(c): int((y == int(c)).sum()) for c in classes}
        w_by_class = {c: (1.0 / math.sqrt(max(1, f))) for c, f in freq.items()}
        mean_w = float(np.mean(list(w_by_class.values()))) if w_by_class else 1.0
        if mean_w > 0:
            w_by_class = {c: (w / mean_w) for c, w in w_by_class.items()}
    else:
        raise ValueError(f"Unknown class balancing mode: {mode}")

    return np.asarray([w_by_class[int(label)] for label in y], dtype="float64")


def build_pipeline(
    *,
    cfg: CallCenterConfig,
    spec: LinearSvcFeatureSpec,
    word_ngram_range: tuple[int, int],
    char_ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    max_features_word: int,
    max_features_char: int,
    sublinear_tf: bool,
    classifier: str,
    clf_params: dict[str, Any] | None,
    seed: int,
) -> Pipeline:
    """Create a pipeline that is safe to joblib.dump/load."""

    clf_params = dict(clf_params or {})

    pre = ColumnTransformer(
        transformers=[
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(char_ngram_range),
                    min_df=int(min_df),
                    max_df=float(max_df),
                    max_features=int(max_features_char),
                    sublinear_tf=bool(sublinear_tf),
                    strip_accents=None,
                    lowercase=False,
                ),
                str(spec.text_col),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(word_ngram_range),
                    min_df=int(min_df),
                    max_df=float(max_df),
                    max_features=int(max_features_word),
                    sublinear_tf=bool(sublinear_tf),
                    strip_accents=None,
                    lowercase=False,
                ),
                str(spec.text_col),
            ),
            ("cats", OneHotEncoder(handle_unknown="ignore"), list(spec.cat_cols)),
            ("num", Pipeline([("scale", MaxAbsScaler())]), list(spec.num_cols)),
        ],
        remainder="drop",
    )

    clf_name = str(classifier).lower().strip()
    if clf_name == "linearsvc":
        clf = LinearSVC(
            C=float(clf_params.get("C", 0.5)),
            max_iter=int(clf_params.get("max_iter", 12000)),
            random_state=int(seed),
        )
    elif clf_name == "logreg":
        penalty = str(clf_params.get("penalty", "l2"))
        clf = LogisticRegression(
            C=float(clf_params.get("C", 1.0)),
            penalty=penalty,
            l1_ratio=float(clf_params.get("l1_ratio", 0.1)) if penalty == "elasticnet" else None,
            solver="saga",
            multi_class="auto",
            max_iter=int(clf_params.get("max_iter", 4000)),
            n_jobs=int(clf_params.get("n_jobs", -1)),
            random_state=int(seed),
        )
    elif clf_name == "sgd":
        clf = SGDClassifier(
            loss=str(clf_params.get("loss", "log_loss")),
            alpha=float(clf_params.get("alpha", 1e-5)),
            penalty=str(clf_params.get("penalty", "l2")),
            l1_ratio=float(clf_params.get("l1_ratio", 0.15)),
            max_iter=int(clf_params.get("max_iter", 2000)),
            tol=float(clf_params.get("tol", 1e-3)),
            random_state=int(seed),
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return Pipeline(
        [
            ("feat", FeatureEngineer(cfg=cfg, spec=spec)),
            ("pre", pre),
            ("clf", clf),
        ]
    )


def cv_score_macro_f1(
    *,
    cfg: CallCenterConfig,
    df_clean: pd.DataFrame,
    y: np.ndarray,
    pipeline: Pipeline,
    n_splits: int,
    seed: int,
    class_balance: str,
) -> dict[str, Any]:
    """StratifiedKFold CV, returning OOF macro-F1 and per-class F1."""

    from sklearn.metrics import f1_score

    y = np.asarray(y).astype(int)
    labels = sorted(np.unique(y).tolist())
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))

    fold_scores: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    oof_pred = np.full(shape=(len(df_clean),), fill_value=-1, dtype=int)

    # basic leakage/quality checks
    id_dups = int(df_clean["id"].duplicated().sum()) if "id" in df_clean.columns else 0
    text_dups = int(
        df_clean[cfg.text_col].astype("string").fillna("").duplicated().sum()
    ) if cfg.text_col in df_clean.columns else 0

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_clean, y), start=1):
        df_tr, df_va = df_clean.iloc[tr_idx], df_clean.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        sw = compute_sample_weight(y_tr, mode=class_balance)
        fit_params = {"clf__sample_weight": sw} if sw is not None else {}

        model = pipeline
        model.fit(df_tr, y_tr, **fit_params)
        pred = model.predict(df_va).astype(int)
        oof_pred[va_idx] = pred

        f = float(f1_score(y_va, pred, average="macro", labels=labels, zero_division=0))
        fold_scores.append(f)
        fold_rows.append({"fold": fold, "macro_f1": f, "n_train": int(len(tr_idx)), "n_valid": int(len(va_idx))})

    oof = float(f1_score(y, oof_pred, average="macro", labels=labels, zero_division=0))
    report = classification_report(y, oof_pred, labels=labels, output_dict=True, zero_division=0)
    per_class_f1 = {str(k): float(v["f1-score"]) for k, v in report.items() if str(k).isdigit()}

    return {
        "labels": labels,
        "folds": fold_rows,
        "macro_f1_mean": float(np.mean(fold_scores)) if fold_scores else 0.0,
        "macro_f1_std": float(np.std(fold_scores)) if fold_scores else 0.0,
        "oof_macro_f1": float(oof),
        "per_class_f1": per_class_f1,
        "classification_report": report,
        "sanity": {"duplicate_id_rows": id_dups, "duplicate_text_rows": text_dups},
    }
