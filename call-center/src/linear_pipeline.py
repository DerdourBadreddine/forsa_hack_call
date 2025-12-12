"""Unified linear text+structured pipeline for Call Center classification.

This is the "final path" requested for FORSA 2025:
- No torch/transformers dependency
- TF-IDF (word + char_wb) over a single document string per ticket
- Optional structured tokens: COL=VALUE
- Optional coarse time tokens from handle_time

The pipeline is intentionally explicit and reproducible.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

import preprocess
from config import CallCenterConfig, default_config, guess_repo_root


_TOKEN_SANITIZE_RE = re.compile(r"\s+")


def _sanitize_token_value(v: str) -> str:
    v = _TOKEN_SANITIZE_RE.sub("_", str(v).strip())
    if not v:
        return "UNK"
    return v


@dataclass(frozen=True)
class DocumentConfig:
    # Which canonical columns to convert to tokens: COL=VALUE
    structured_cols: tuple[str, ...] = (
        "tt_status",
        "dept_initial",
        "service_request_type",
        "subs_level",
        "customer_level",
        "dept_traitement",
        "tech_comm",
        "service_type",
        "dot",
        "actel",
        "actel_code",
    )

    # Whether to add coarse time tokens from handle_time
    add_time_tokens: bool = False

    # Whether to prepend structured tokens into the same text string
    add_structured_tokens: bool = True

    # Whether to include the cleaned free text at all (normally True)
    add_text: bool = True


def make_document_series(
    df_clean: pd.DataFrame,
    cfg: CallCenterConfig,
    doc_cfg: DocumentConfig,
) -> pd.Series:
    """Build one document string per row from a cleaned canonical dataframe."""

    parts: list[pd.Series] = []

    if doc_cfg.add_structured_tokens:
        for col in doc_cfg.structured_cols:
            if col not in df_clean.columns:
                continue
            s = df_clean[col].astype("string").fillna("UNK").map(_sanitize_token_value)
            token = f"{col.upper()}="
            parts.append(token + s)

    if doc_cfg.add_time_tokens:
        # Use very coarse bins only.
        if cfg.datetime_col in df_clean.columns:
            dt = pd.to_datetime(df_clean[cfg.datetime_col], errors="coerce")
            hour = dt.dt.hour.astype("Int64").astype("string").fillna("UNK")
            dow = dt.dt.dayofweek.astype("Int64").astype("string").fillna("UNK")
            month = dt.dt.month.astype("Int64").astype("string").fillna("UNK")
            parts.append("HOUR=" + hour)
            parts.append("DOW=" + dow)
            parts.append("MONTH=" + month)

    if doc_cfg.add_text:
        if cfg.text_col not in df_clean.columns:
            raise KeyError(f"Missing text column '{cfg.text_col}' in cleaned dataframe")
        parts.append(df_clean[cfg.text_col].astype("string").fillna("UNK"))

    if not parts:
        return pd.Series(["UNK"] * len(df_clean), index=df_clean.index, dtype="string")

    doc = parts[0]
    for p in parts[1:]:
        doc = doc + " " + p
    return doc.astype("string")


def preprocess_raw_df(df_raw: pd.DataFrame, cfg: CallCenterConfig) -> pd.DataFrame:
    """Raw -> cleaned canonical (uses existing preprocessing)."""
    # Some competitions use 'Id'/'Prediction' casing; normalize gently.
    df = df_raw.copy()
    if "Id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"Id": "id"})
    if "Prediction" in df.columns and "class_int" not in df.columns:
        df = df.rename(columns={"Prediction": "class_int"})

    return preprocess.preprocess_callcenter_df(df, cfg)


def build_unified_pipeline(
    *,
    cfg: CallCenterConfig | None = None,
    doc_cfg: DocumentConfig | None = None,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features_word: int = 120_000,
    max_features_char: int = 250_000,
    sublinear_tf: bool = True,
    classifier: str = "linearsvc",  # {logreg,linearsvc,sgd}
    clf_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> Pipeline:
    """Create the sklearn Pipeline.

    Pipeline layout:
      df(cleaned canonical) -> doc string -> (word tfidf + char tfidf) -> classifier
    """

    cfg = cfg or default_config()
    doc_cfg = doc_cfg or DocumentConfig()
    clf_params = dict(clf_params or {})

    def _to_docs(frame: pd.DataFrame) -> np.ndarray:
        docs = make_document_series(frame, cfg, doc_cfg)
        return docs.to_numpy(dtype=object)

    doc_builder = FunctionTransformer(_to_docs, validate=False)

    # Two TF-IDF vectorizers over the same document string.
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(word_ngram_range),
        min_df=int(min_df),
        max_df=float(max_df),
        max_features=int(max_features_word),
        sublinear_tf=bool(sublinear_tf),
        strip_accents=None,
        lowercase=False,  # text already cleaned
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(char_ngram_range),
        min_df=int(min_df),
        max_df=float(max_df),
        max_features=int(max_features_char),
        sublinear_tf=bool(sublinear_tf),
        strip_accents=None,
        lowercase=False,
    )

    # We use FeatureUnion via ColumnTransformer-like trick: just concatenate sparse matrices.
    # sklearn doesn't have a native "two vectorizers on same input" primitive other than FeatureUnion.
    from sklearn.pipeline import FeatureUnion

    feats = FeatureUnion([("word", word_vec), ("char", char_vec)])

    clf_name = str(classifier).lower().strip()
    if clf_name == "linearsvc":
        clf = LinearSVC(
            C=float(clf_params.get("C", 1.0)),
            max_iter=int(clf_params.get("max_iter", 12000)),
            random_state=int(seed),
        )
    elif clf_name == "logreg":
        penalty = str(clf_params.get("penalty", "l2"))
        solver = "saga"
        clf = LogisticRegression(
            C=float(clf_params.get("C", 1.0)),
            penalty=penalty,
            l1_ratio=float(clf_params.get("l1_ratio", 0.1)) if penalty == "elasticnet" else None,
            solver=solver,
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

    return Pipeline([
        ("doc", doc_builder),
        ("tfidf", feats),
        ("clf", clf),
    ])


def compute_sample_weight(y: np.ndarray, *, mode: str) -> np.ndarray | None:
    """Compute sample_weight for imbalance handling.

    mode:
      - 'none'
      - 'balanced' (sklearn's balanced weights)
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
        # normalize to mean 1 for numerical stability
        mean_w = float(np.mean(list(w_by_class.values()))) if w_by_class else 1.0
        if mean_w > 0:
            w_by_class = {c: (w / mean_w) for c, w in w_by_class.items()}
    else:
        raise ValueError(f"Unknown class balancing mode: {mode}")

    return np.asarray([w_by_class[int(label)] for label in y], dtype="float64")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int] | None = None) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def cv_score_macro_f1(
    *,
    df_clean: pd.DataFrame,
    y: np.ndarray,
    pipeline: Pipeline,
    n_splits: int,
    seed: int,
    class_balance: str,
) -> dict[str, Any]:
    """StratifiedKFold CV returning mean/std macro-F1 and per-fold details."""

    y = np.asarray(y).astype(int)
    labels = sorted(np.unique(y).tolist())
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))

    fold_scores: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    oof_pred = np.full(shape=(len(df_clean),), fill_value=-1, dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_clean, y), start=1):
        df_tr, df_va = df_clean.iloc[tr_idx], df_clean.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        sw = compute_sample_weight(y_tr, mode=class_balance)
        fit_params = {"clf__sample_weight": sw} if sw is not None else {}

        model = pipeline
        model.fit(df_tr, y_tr, **fit_params)
        pred = model.predict(df_va).astype(int)
        oof_pred[va_idx] = pred

        f = macro_f1(y_va, pred, labels=labels)
        fold_scores.append(float(f))
        fold_rows.append({"fold": fold, "macro_f1": float(f), "n_train": int(len(tr_idx)), "n_valid": int(len(va_idx))})

    oof = macro_f1(y, oof_pred, labels=labels)

    from sklearn.metrics import classification_report

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
    }


def default_local_data_dir() -> Path:
    return guess_repo_root() / "data" / "forsa-2025-call-center"


def default_outputs_dir() -> Path:
    return guess_repo_root() / "outputs" / "callcenter" / "linear"


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
