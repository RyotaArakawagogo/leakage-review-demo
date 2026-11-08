"""Evaluate one model with an image-level split and with a patient-level split."""
from __future__ import annotations

from typing import Iterator, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import FEATURE_NAMES

Scheme = Literal["image", "patient"]
N_SPLITS = 5
SPLIT_SEED = 0
MODEL_SEED = 0


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=MODEL_SEED, n_jobs=1)


def split_indices(df: pd.DataFrame, scheme: Scheme, n_splits: int = N_SPLITS) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    X = df[FEATURE_NAMES].to_numpy()
    y = df["label"].to_numpy()
    if scheme == "image":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SPLIT_SEED).split(X, y)
    if scheme == "patient":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SPLIT_SEED)
        return cv.split(X, y, groups=df["patient_id"].to_numpy())
    raise ValueError(f"unknown scheme: {scheme}")


def shared_patient_count(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> int:
    pid = df["patient_id"].to_numpy()
    return len(set(pid[train_idx]) & set(pid[test_idx]))


def evaluate(df: pd.DataFrame, scheme: Scheme, n_splits: int = N_SPLITS) -> dict:
    X = df[FEATURE_NAMES].to_numpy()
    y = df["label"].to_numpy()
    folds = []
    # Out-of-fold predictions: each image's probability from the fold where it was in test.
    oof_proba = np.full(len(df), np.nan)
    oof_fold = np.full(len(df), -1)
    for fold, (train_idx, test_idx) in enumerate(split_indices(df, scheme, n_splits)):
        model = make_model().fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        oof_proba[test_idx] = proba
        oof_fold[test_idx] = fold
        folds.append(
            {
                "auc": roc_auc_score(y[test_idx], proba),
                "accuracy": accuracy_score(y[test_idx], (proba >= 0.5).astype(int)),
                "shared_patients": shared_patient_count(df, train_idx, test_idx),
                "test_patients": df["patient_id"].iloc[test_idx].nunique(),
            }
        )
    auc = np.array([f["auc"] for f in folds])
    acc = np.array([f["accuracy"] for f in folds])
    return {
        "scheme": scheme,
        "folds": folds,
        "oof_proba": oof_proba,
        "oof_fold": oof_fold,
        # Sample standard deviation over the folds (ddof=1).
        "auc_mean": float(auc.mean()),
        "auc_std": float(auc.std(ddof=1)),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
    }
