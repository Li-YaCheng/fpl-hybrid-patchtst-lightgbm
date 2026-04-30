from __future__ import annotations

import numpy as np


def threshold_grid() -> np.ndarray:
    # Paper: [0.03, 0.97], step 0.005
    return np.round(np.arange(0.03, 0.97001, 0.005), 3)


def best_thr_by_val_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    from sklearn.metrics import f1_score  # type: ignore

    y_val = np.asarray(y_val).astype(int).ravel()
    p_val = np.asarray(p_val).astype(float).ravel()
    best_thr, best = 0.5, -1.0
    for thr in threshold_grid():
        f1 = f1_score(y_val, (p_val >= thr).astype(int))
        if f1 > best:
            best = float(f1)
            best_thr = float(thr)
    return float(best_thr)


def metrics_at_thr(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # type: ignore

    y_true = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    y_pred = (p >= float(thr)).astype(int)
    return {
        "F1": float(f1_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, p)),
        "Prec.": float(precision_score(y_true, y_pred, zero_division=0)),
        "Rec.": float(recall_score(y_true, y_pred, zero_division=0)),
    }

