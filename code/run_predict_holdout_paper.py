from __future__ import annotations

"""
Paper-aligned holdout (2025–26) prediction for the Hybrid model.

Implements Section 6 of ACM双栏.pdf:
- Train on historical seasons (2016–2025) and predict 2025–26 holdout.
- Stage-1: PatchTST-lite ensemble (seeds 42,43,44) -> mu_p, sigma_p
- Stage-2: LightGBM fusion on Z=[flat(X); mu_p; sigma_p]

Outputs:
  - artifacts/pred/predict_25_26_hybrid_predictions_patchtst_paper.csv
  - data/predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv  (legacy filename used by figure scripts)

Notes:
- This script assumes sequences are leakage-free (see `sequence_generator_paper20d.py`).
"""

import csv
import math
from pathlib import Path

import numpy as np

from patchtst_paper import PatchTSTPaperConfig, fit_predict_all


PKG_ROOT = Path(__file__).resolve().parents[1]
FAST = (PKG_ROOT / ".fast").exists()


def _read_meta(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            rows.append(rr)
    return rows


def _train_lgbm(Z_tr: np.ndarray, y_tr: np.ndarray, Z_va: np.ndarray, y_va: np.ndarray, *, seed: int):
    import lightgbm as lgb  # type: ignore

    dtr = lgb.Dataset(Z_tr, label=y_tr)
    dva = lgb.Dataset(Z_va, label=y_va, reference=dtr)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": int(seed),
        "feature_pre_filter": False,
        "is_unbalance": True,
    }
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=200 if FAST else 600,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(15 if FAST else 50, verbose=False)],
    )
    best_iter = int(getattr(booster, "best_iteration", 0) or 0)
    return booster, best_iter


def main() -> None:
    npz_path = PKG_ROOT / "artifacts" / "npz" / "paper20d_L30.npz"
    meta_path = PKG_ROOT / "data" / "processed" / "paper20d_sequences_holdout_meta_L30.csv"
    if not npz_path.exists():
        raise SystemExit(f"Missing: {npz_path}. Run data_loader_paper20d.py + sequence_generator_paper20d.py first.")
    if not meta_path.exists():
        raise SystemExit(f"Missing: {meta_path}. Run sequence_generator_paper20d.py first.")

    z = np.load(npz_path, allow_pickle=True)
    X_tr, y_tr = z["X_train"].astype(np.float32), z["y_train"].astype(int)
    X_va, y_va = z["X_val"].astype(np.float32), z["y_val"].astype(int)
    X_te, y_te = z["X_test"].astype(np.float32), z["y_test"].astype(int)
    X_pred = z["X_predict"].astype(np.float32)
    y_pred = z["y_predict"].astype(int)

    meta = _read_meta(meta_path)
    if len(meta) != len(X_pred):
        raise SystemExit(f"Meta rows ({len(meta)}) != X_predict ({len(X_pred)}).")

    # Deployment: train on historical 2016–2025 (train+val+test).
    X_hist = np.concatenate([X_tr, X_va, X_te], axis=0)
    y_hist = np.concatenate([y_tr, y_va, y_te], axis=0)

    # Keep val split for early-stopping / threshold selection behavior (paper uses Val year).
    X_val = X_va
    y_val = y_va

    # Stage-1 PatchTST ensemble -> mu/sigma for hist, val, holdout
    seeds = [42, 43, 44]
    cfg = PatchTSTPaperConfig(max_epochs=10 if FAST else 60)

    p_hist_list: list[np.ndarray] = []
    p_val_list: list[np.ndarray] = []
    p_hold_list: list[np.ndarray] = []

    # We train each seed model on the historical set and predict hist/val/holdout.
    # (val is a subset of hist but kept for protocol consistency.)
    for s in seeds:
        p_tr_s, _e_tr, p_va_s, _e_va, p_te_s, _e_te = fit_predict_all(X_hist, y_hist, X_val, y_val, X_pred, seed=int(s), cfg=cfg)
        p_hist_list.append(p_tr_s)
        p_val_list.append(p_va_s)
        p_hold_list.append(p_te_s)

    P_hist = np.stack(p_hist_list, axis=0)
    P_val = np.stack(p_val_list, axis=0)
    P_hold = np.stack(p_hold_list, axis=0)

    mu_hist, sig_hist = P_hist.mean(axis=0), P_hist.std(axis=0, ddof=0)
    mu_val, sig_val = P_val.mean(axis=0), P_val.std(axis=0, ddof=0)
    mu_hold, sig_hold = P_hold.mean(axis=0), P_hold.std(axis=0, ddof=0)

    # Stage-2 LightGBM fusion on Z=[flat(X); mu; sigma]
    X_hist_flat = X_hist.reshape((len(X_hist), -1))
    X_val_flat = X_val.reshape((len(X_val), -1))
    X_hold_flat = X_pred.reshape((len(X_pred), -1))

    Z_hist = np.concatenate([X_hist_flat, mu_hist.reshape(-1, 1), sig_hist.reshape(-1, 1)], axis=1)
    Z_val = np.concatenate([X_val_flat, mu_val.reshape(-1, 1), sig_val.reshape(-1, 1)], axis=1)
    Z_hold = np.concatenate([X_hold_flat, mu_hold.reshape(-1, 1), sig_hold.reshape(-1, 1)], axis=1)

    booster, best_iter = _train_lgbm(Z_hist, y_hist, Z_val, y_val, seed=42)
    p_hold = np.asarray(booster.predict(Z_hold, num_iteration=best_iter or None), dtype=float).reshape(-1)

    # Write legacy prediction CSV for downstream figure scripts (expects hybrid_proba)
    out_pred = PKG_ROOT / "artifacts" / "pred" / "predict_25_26_hybrid_predictions_patchtst_paper.csv"
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_legacy = PKG_ROOT / "data" / "predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv"

    header = ["player_name", "season_x", "round", "kickoff_time", "total_points", "y_true", "hybrid_proba"]

    def write(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for m, p, yy in zip(meta, p_hold, y_pred):
                name = (m.get("player_name") or "").strip()
                season = (m.get("season") or "2025-26").strip()
                rd = int(float(m.get("round", "0")))
                ko = (m.get("kickoff_time") or "").strip()
                tp = float(m.get("total_points", "nan")) if m.get("total_points", "") != "" else float("nan")
                yv = m.get("y_true", "")
                if yv == "":
                    yv = int(yy) if int(yy) >= 0 else ""
                w.writerow([name, season, rd, ko, tp if math.isfinite(tp) else "", yv, float(p)])

    write(out_pred)
    write(out_legacy)

    print("Wrote:", out_pred)
    print("Wrote:", out_legacy)


if __name__ == "__main__":
    main()

