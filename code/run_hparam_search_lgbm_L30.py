from __future__ import annotations

"""
Hyperparameter search for LightGBM(flat sequence) at seq_len=30 (reconstructed).

Inputs:
  - artifacts/npz/processed_data_v2_L30_padpred.npz

Outputs:
  - tables/lgbm_flat_L30_sweep_results.csv
  - tables/lgbm_flat_L30_sweep_best.json

Protocol:
- Train on X_train, tune on X_val (AUC for early stopping; F1 via threshold tuning on val)
- Report both val and test metrics for each trial (threshold fixed from val)
"""

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrialResult:
    trial: int
    num_leaves: int
    min_data_in_leaf: int
    learning_rate: float
    feature_fraction: float
    bagging_fraction: float
    lambda_l2: float
    best_iter: int
    val_auc: float
    val_f1: float
    test_auc: float
    test_f1: float
    thr: float
    train_time_s: float


def _best_thr_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score  # type: ignore

    best_thr, best = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (p >= thr).astype(int))
        if f1 > best:
            best = f1
            best_thr = float(thr)
    return best_thr


def main() -> None:
    try:
        import lightgbm as lgb  # type: ignore
        from sklearn.metrics import roc_auc_score, f1_score  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Dependencies missing. Use upload/environment.yml (numpy<2) to run.\n"
            f"Import error: {e}"
        )

    npz_path = PKG_ROOT / "artifacts" / "npz" / "processed_data_v2_L30_padpred.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing: {npz_path}. Run sequence_generator_15dim.py first.")

    z = np.load(npz_path, allow_pickle=True)
    X_tr, y_tr = z["X_train"].astype(np.float32), z["y_train"].astype(int)
    X_va, y_va = z["X_val"].astype(np.float32), z["y_val"].astype(int)
    X_te, y_te = z["X_test"].astype(np.float32), z["y_test"].astype(int)

    Xtr = X_tr.reshape((len(X_tr), -1))
    Xva = X_va.reshape((len(X_va), -1))
    Xte = X_te.reshape((len(X_te), -1))

    out_csv = PKG_ROOT / "tables" / "lgbm_flat_L30_sweep_results.csv"
    out_best = PKG_ROOT / "tables" / "lgbm_flat_L30_sweep_best.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    def sample_params() -> dict:
        return {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 42,
            # Required when tuning min_data_in_leaf across trials; otherwise LightGBM can error.
            "feature_pre_filter": False,
            "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05])),
            "num_leaves": int(rng.choice([31, 63, 127, 255])),
            "min_data_in_leaf": int(rng.choice([30, 50, 80, 120])),
            "feature_fraction": float(rng.choice([0.75, 0.85, 0.9, 1.0])),
            "bagging_fraction": float(rng.choice([0.75, 0.85, 0.9, 1.0])),
            "bagging_freq": 1,
            "lambda_l2": float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
        }

    trials = 24
    results: list[TrialResult] = []

    dtr = lgb.Dataset(Xtr, label=y_tr)
    dva = lgb.Dataset(Xva, label=y_va, reference=dtr)

    for t in range(1, trials + 1):
        params = sample_params()
        t0 = time.time()
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=3000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        best_iter = int(getattr(booster, "best_iteration", 0) or 0)
        pva = booster.predict(Xva, num_iteration=best_iter or None)
        thr = _best_thr_f1(y_va, pva)
        pte = booster.predict(Xte, num_iteration=best_iter or None)
        res = TrialResult(
            trial=t,
            num_leaves=int(params["num_leaves"]),
            min_data_in_leaf=int(params["min_data_in_leaf"]),
            learning_rate=float(params["learning_rate"]),
            feature_fraction=float(params["feature_fraction"]),
            bagging_fraction=float(params["bagging_fraction"]),
            lambda_l2=float(params["lambda_l2"]),
            best_iter=best_iter,
            val_auc=float(roc_auc_score(y_va, pva)),
            val_f1=float(f1_score(y_va, (pva >= thr).astype(int))),
            test_auc=float(roc_auc_score(y_te, pte)),
            test_f1=float(f1_score(y_te, (pte >= thr).astype(int))),
            thr=float(thr),
            train_time_s=float(time.time() - t0),
        )
        results.append(res)
        print(f"trial {t:02d}: val_f1={res.val_f1:.4f} test_f1={res.test_f1:.4f} iter={res.best_iter}")

    # sort by val_f1 desc
    results.sort(key=lambda r: r.val_f1, reverse=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    best = asdict(results[0])
    out_best.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Wrote:", out_csv)
    print("Best:", out_best)


if __name__ == "__main__":
    main()

