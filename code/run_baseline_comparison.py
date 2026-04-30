"""
Baseline / horizontal comparison (reconstructed).

This script is intended to match the role of the original `run_baseline_comparison.py`:
- Load sequence tensors from `artifacts/npz/processed_data_v2_L30_padpred.npz`
- Train a small set of baselines on the same split
- Tune threshold on validation set (F1) and report test metrics
- Write a comparison table CSV under `upload/tables/`

The original project used additional models and richer feature pipelines; since the original scripts
were deleted and cannot be recovered, this is a compatible reconstruction.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]
FAST = (PKG_ROOT / ".fast").exists()
INCLUDE_LOGREG = os.environ.get("INCLUDE_LOGREG", "").strip() in {"1", "true", "True", "YES", "yes"}


@dataclass
class EvalResult:
    model: str
    f1: float
    auc: float
    accuracy: float
    precision: float
    recall: float
    threshold: float
    train_time_s: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _eval_binary(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore

    y_pred = (y_proba >= thr).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "auc": _safe_auc(y_true, y_proba),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _find_best_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = _eval_binary(y_true, y_proba, float(thr))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _flatten_last_step(X: np.ndarray) -> np.ndarray:
    # (N,L,D) -> (N,D) using last step
    return X[:, -1, :]


def _flatten_all(X: np.ndarray) -> np.ndarray:
    # (N,L,D) -> (N,L*D)
    return X.reshape((X.shape[0], -1))


def _sanitize_X(X: np.ndarray) -> np.ndarray:
    # sklearn's LogisticRegression does not accept NaN/Inf.
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    npz_path = PKG_ROOT / "artifacts" / "npz" / "processed_data_v2_L30_padpred.npz"
    out_csv = PKG_ROOT / "tables" / "baseline_comparison_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        raise SystemExit(f"Missing: {npz_path}. Run sequence_generator_15dim.py first.")

    z = np.load(npz_path, allow_pickle=True)
    X_tr, y_tr = z["X_train"].astype(np.float32), z["y_train"].astype(int)
    X_va, y_va = z["X_val"].astype(np.float32), z["y_val"].astype(int)
    X_te, y_te = z["X_test"].astype(np.float32), z["y_test"].astype(int)

    # Baselines (requires sklearn)
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
    except Exception as e:
        raise SystemExit(
            "scikit-learn not available. Please create the clean env (numpy<2) from upload/environment.yml.\n"
            f"Import error: {e}"
        )

    results: list[EvalResult] = []

    # Logistic regression on flattened sequence
    if INCLUDE_LOGREG:
        for name, fe in [
            ("LogReg(last-step)", _flatten_last_step),
            ("LogReg(flat)", _flatten_all),
        ]:
            t0 = time.time()
            Xtr = fe(X_tr)
            Xva = fe(X_va)
            Xte = fe(X_te)
            Xtr = _sanitize_X(Xtr)
            Xva = _sanitize_X(Xva)
            Xte = _sanitize_X(Xte)
            clf = LogisticRegression(max_iter=300)
            clf.fit(Xtr, y_tr)
            pva = clf.predict_proba(Xva)[:, 1]
            thr = _find_best_threshold_by_f1(y_va, pva)
            pte = clf.predict_proba(Xte)[:, 1]
            m = _eval_binary(y_te, pte, thr)
            results.append(
                EvalResult(
                    model=name,
                    threshold=thr,
                    train_time_s=float(time.time() - t0),
                    f1=m["f1"],
                    auc=m["auc"],
                    accuracy=m["accuracy"],
                    precision=m["precision"],
                    recall=m["recall"],
                )
            )

    # Random forest on last-step (fast baseline)
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=50 if FAST else 300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(_sanitize_X(_flatten_last_step(X_tr)), y_tr)
    pva = rf.predict_proba(_sanitize_X(_flatten_last_step(X_va)))[:, 1]
    thr = _find_best_threshold_by_f1(y_va, pva)
    pte = rf.predict_proba(_sanitize_X(_flatten_last_step(X_te)))[:, 1]
    m = _eval_binary(y_te, pte, thr)
    results.append(
        EvalResult(
            model="RF(last-step)",
            threshold=thr,
            train_time_s=float(time.time() - t0),
            f1=m["f1"],
            auc=m["auc"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
        )
    )

    # LightGBM on last-step (strong tabular baseline)
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        raise SystemExit(f"LightGBM missing: {e}")

    t0 = time.time()
    Xtr = _sanitize_X(_flatten_last_step(X_tr))
    Xva = _sanitize_X(_flatten_last_step(X_va))
    Xte = _sanitize_X(_flatten_last_step(X_te))
    dtr = lgb.Dataset(Xtr, label=y_tr)
    dva = lgb.Dataset(Xva, label=y_va, reference=dtr)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
    }
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=600,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = int(getattr(booster, "best_iteration", 0) or 0)
    pva = booster.predict(Xva, num_iteration=best_iter or None)
    thr = _find_best_threshold_by_f1(y_va, pva)
    pte = booster.predict(Xte, num_iteration=best_iter or None)
    m = _eval_binary(y_te, pte, thr)
    results.append(
        EvalResult(
            model="LightGBM(last-step)",
            threshold=thr,
            train_time_s=float(time.time() - t0),
            f1=m["f1"],
            auc=m["auc"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
        )
    )

    # LightGBM on full flattened sequence (this is "LightGBM (flat L=30)" row in the paper table)
    t0 = time.time()
    Xtr_f = _sanitize_X(_flatten_all(X_tr))
    Xva_f = _sanitize_X(_flatten_all(X_va))
    Xte_f = _sanitize_X(_flatten_all(X_te))
    dtr_f = lgb.Dataset(Xtr_f, label=y_tr)
    dva_f = lgb.Dataset(Xva_f, label=y_va, reference=dtr_f)
    booster_f = lgb.train(
        params,
        dtr_f,
        num_boost_round=800 if not FAST else 200,
        valid_sets=[dva_f],
        callbacks=[lgb.early_stopping(50 if not FAST else 20, verbose=False)],
    )
    best_iter_f = int(getattr(booster_f, "best_iteration", 0) or 0)
    pva_f = booster_f.predict(Xva_f, num_iteration=best_iter_f or None)
    thr_f = _find_best_threshold_by_f1(y_va, pva_f)
    pte_f = booster_f.predict(Xte_f, num_iteration=best_iter_f or None)
    m_f = _eval_binary(y_te, pte_f, thr_f)
    results.append(
        EvalResult(
            model="LightGBM(flat L=30)",
            threshold=thr_f,
            train_time_s=float(time.time() - t0),
            f1=m_f["f1"],
            auc=m_f["auc"],
            accuracy=m_f["accuracy"],
            precision=m_f["precision"],
            recall=m_f["recall"],
        )
    )

    # XGBoost on last-step (tabular baseline)
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception:
        XGBClassifier = None

    if XGBClassifier is not None:
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=400 if not FAST else 120,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
        )
        xgb.fit(Xtr, y_tr)
        pva = xgb.predict_proba(Xva)[:, 1]
        thr = _find_best_threshold_by_f1(y_va, pva)
        pte = xgb.predict_proba(Xte)[:, 1]
        m = _eval_binary(y_te, pte, thr)
        results.append(
            EvalResult(
                model="XGBoost(last-step)",
                threshold=thr,
                train_time_s=float(time.time() - t0),
                f1=m["f1"],
                auc=m["auc"],
                accuracy=m["accuracy"],
                precision=m["precision"],
                recall=m["recall"],
            )
        )

    # Write CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "f1",
                "auc",
                "accuracy",
                "precision",
                "recall",
                "threshold",
                "train_time_s",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in asdict(r).items()})

    (PKG_ROOT / "tables" / "baseline_comparison_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()

