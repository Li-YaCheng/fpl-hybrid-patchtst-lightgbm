from __future__ import annotations

"""
Paper-aligned Table 5: Cross-Model Comparison Results.

Implements Table 5 objects and evaluation protocol from ACM双栏.pdf:
- Data splits: Train 2016–22, Val 2022–23, Test 2023–25
- Input: 20-D per timestep, L=30
- Threshold τ* selected on Val by maximizing Val-F1 over grid [0.03,0.97], step 0.005

Rows (exact objects):
1 Ours (L=30, Hybrid)          = PatchTST-lite ensemble -> (mu_p, sigma_p) + LightGBM fusion on [flat(X); mu_p; sigma_p]
2 LightGBM                     = LightGBM on last-step (t0) 20-D features
3 LightGBM (flat L=30)         = LightGBM on flattened 600-D temporal features
4 XGBoost                      = XGBoost on last-step 20-D features
5 Random Forest                = RandomForest on last-step 20-D features
6 PatchTST (end-to-end)        = PatchTST-lite single model with direct classification head (no LightGBM stage-2)

Outputs:
  - tables/table5_cross_model_comparison.csv
  - tables/table5_cross_model_comparison.json (debug)
"""

import csv
import json
import time
from pathlib import Path

import numpy as np

from paper_eval import best_thr_by_val_f1, metrics_at_thr
from patchtst_paper import PatchTSTPaperConfig, fit_predict_all


PKG_ROOT = Path(__file__).resolve().parents[1]
FAST = (PKG_ROOT / ".fast").exists()


def _load_npz() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz = PKG_ROOT / "artifacts" / "npz" / "paper20d_L30.npz"
    if not npz.exists():
        raise SystemExit(f"Missing: {npz}. Run data_loader_paper20d.py + sequence_generator_paper20d.py first.")
    z = np.load(npz, allow_pickle=True)
    return (
        z["X_train"].astype(np.float32),
        z["y_train"].astype(int),
        z["X_val"].astype(np.float32),
        z["y_val"].astype(int),
        z["X_test"].astype(np.float32),
        z["y_test"].astype(int),
    )


def _train_lgbm(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, *, seed: int) -> tuple[object, int]:
    import lightgbm as lgb  # type: ignore

    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
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


def _predict_lgbm(booster: object, X: np.ndarray, best_iter: int) -> np.ndarray:
    return np.asarray(booster.predict(X, num_iteration=best_iter or None), dtype=float).reshape(-1)


def _train_xgb(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, *, seed: int):
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing xgboost. Install it in your env. Import error: {e}")

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": int(seed),
    }
    num_round = 200 if FAST else 800
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=num_round,
        evals=[(dva, "val")],
        verbose_eval=False,
        early_stopping_rounds=15 if FAST else 50,
    )
    return bst


def main() -> None:
    X_tr, y_tr, X_va, y_va, X_te, y_te = _load_npz()

    # Flattened views
    Xtr_flat = X_tr.reshape((len(X_tr), -1))
    Xva_flat = X_va.reshape((len(X_va), -1))
    Xte_flat = X_te.reshape((len(X_te), -1))
    # Last-step views (t0)
    Xtr_t0 = X_tr[:, -1, :].reshape((len(X_tr), -1))
    Xva_t0 = X_va[:, -1, :].reshape((len(X_va), -1))
    Xte_t0 = X_te[:, -1, :].reshape((len(X_te), -1))

    out_dir = PKG_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "table5_cross_model_comparison.csv"
    out_json = out_dir / "table5_cross_model_comparison.json"

    rows_debug: list[dict] = []

    # === 1) Ours (Hybrid) ===
    t0 = time.time()
    seeds = [42, 43, 44]
    cfg = PatchTSTPaperConfig(max_epochs=10 if FAST else 60)
    p_tr_list, p_va_list, p_te_list = [], [], []
    for s in seeds:
        p_tr_s, _emb_tr, p_va_s, _emb_va, p_te_s, _emb_te = fit_predict_all(
            X_tr, y_tr, X_va, y_va, X_te, seed=int(s), cfg=cfg
        )
        p_tr_list.append(p_tr_s)
        p_va_list.append(p_va_s)
        p_te_list.append(p_te_s)

    Ptr = np.stack(p_tr_list, axis=0)
    Pva = np.stack(p_va_list, axis=0)
    Pte = np.stack(p_te_list, axis=0)
    mu_tr, sig_tr = Ptr.mean(axis=0), Ptr.std(axis=0, ddof=0)
    mu_va, sig_va = Pva.mean(axis=0), Pva.std(axis=0, ddof=0)
    mu_te, sig_te = Pte.mean(axis=0), Pte.std(axis=0, ddof=0)

    Ztr = np.concatenate([Xtr_flat, mu_tr.reshape(-1, 1), sig_tr.reshape(-1, 1)], axis=1)
    Zva = np.concatenate([Xva_flat, mu_va.reshape(-1, 1), sig_va.reshape(-1, 1)], axis=1)
    Zte = np.concatenate([Xte_flat, mu_te.reshape(-1, 1), sig_te.reshape(-1, 1)], axis=1)

    booster, best_iter = _train_lgbm(Ztr, y_tr, Zva, y_va, seed=42)
    pva = _predict_lgbm(booster, Zva, best_iter)
    thr = best_thr_by_val_f1(y_va, pva)
    pte = _predict_lgbm(booster, Zte, best_iter)
    m = metrics_at_thr(y_te, pte, thr)
    rows_debug.append(
        {
            "Configuration": "Ours (L=30, Hybrid)",
            **m,
            "ΔF1": "—",
            "_thr": float(thr),
            "_best_iter": int(best_iter),
            "_train_time_s": float(time.time() - t0),
        }
    )

    ours_f1 = float(m["F1"])

    # === 2) LightGBM (t0) ===
    t0 = time.time()
    booster2, bi2 = _train_lgbm(Xtr_t0, y_tr, Xva_t0, y_va, seed=42)
    pva2 = _predict_lgbm(booster2, Xva_t0, bi2)
    thr2 = best_thr_by_val_f1(y_va, pva2)
    pte2 = _predict_lgbm(booster2, Xte_t0, bi2)
    m2 = metrics_at_thr(y_te, pte2, thr2)
    rows_debug.append(
        {
            "Configuration": "LightGBM",
            **m2,
            "ΔF1": f"{(float(m2['F1']) - ours_f1):+.2%}",
            "_thr": float(thr2),
            "_best_iter": int(bi2),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # === 3) LightGBM (flat L=30) ===
    t0 = time.time()
    booster3, bi3 = _train_lgbm(Xtr_flat, y_tr, Xva_flat, y_va, seed=42)
    pva3 = _predict_lgbm(booster3, Xva_flat, bi3)
    thr3 = best_thr_by_val_f1(y_va, pva3)
    pte3 = _predict_lgbm(booster3, Xte_flat, bi3)
    m3 = metrics_at_thr(y_te, pte3, thr3)
    rows_debug.append(
        {
            "Configuration": "LightGBM (flat L=30)",
            **m3,
            "ΔF1": f"{(float(m3['F1']) - ours_f1):+.2%}",
            "_thr": float(thr3),
            "_best_iter": int(bi3),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # === 4) XGBoost (t0) ===
    t0 = time.time()
    bst = _train_xgb(Xtr_t0, y_tr, Xva_t0, y_va, seed=42)
    try:
        import xgboost as xgb  # type: ignore

        pva4 = bst.predict(xgb.DMatrix(Xva_t0))
        thr4 = best_thr_by_val_f1(y_va, pva4)
        pte4 = bst.predict(xgb.DMatrix(Xte_t0))
    except Exception:
        pva4 = np.asarray(bst.predict(Xva_t0), dtype=float).reshape(-1)
        thr4 = best_thr_by_val_f1(y_va, pva4)
        pte4 = np.asarray(bst.predict(Xte_t0), dtype=float).reshape(-1)
    m4 = metrics_at_thr(y_te, pte4, thr4)
    rows_debug.append(
        {
            "Configuration": "XGBoost",
            **m4,
            "ΔF1": f"{(float(m4['F1']) - ours_f1):+.2%}",
            "_thr": float(thr4),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # === 5) Random Forest (t0) ===
    t0 = time.time()
    from sklearn.ensemble import RandomForestClassifier  # type: ignore

    rf = RandomForestClassifier(
        n_estimators=80 if FAST else 400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(Xtr_t0, y_tr)
    pva5 = rf.predict_proba(Xva_t0)[:, 1]
    thr5 = best_thr_by_val_f1(y_va, pva5)
    pte5 = rf.predict_proba(Xte_t0)[:, 1]
    m5 = metrics_at_thr(y_te, pte5, thr5)
    rows_debug.append(
        {
            "Configuration": "Random Forest",
            **m5,
            "ΔF1": f"{(float(m5['F1']) - ours_f1):+.2%}",
            "_thr": float(thr5),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # === 6) PatchTST (end-to-end) ===
    t0 = time.time()
    cfg_e2e = PatchTSTPaperConfig(max_epochs=10 if FAST else 60)
    _ptr6, _embtr6, pva6, _embva6, pte6, _embte6 = fit_predict_all(
        X_tr, y_tr, X_va, y_va, X_te, seed=42, cfg=cfg_e2e
    )
    thr6 = best_thr_by_val_f1(y_va, pva6)
    m6 = metrics_at_thr(y_te, pte6, thr6)
    rows_debug.append(
        {
            "Configuration": "PatchTST (end-to-end)",
            **m6,
            "ΔF1": f"{(float(m6['F1']) - ours_f1):+.2%}",
            "_thr": float(thr6),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # Write CSV (paper columns)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Configuration", "F1", "AUC", "Prec.", "Rec.", "ΔF1"])
        w.writeheader()
        for r in rows_debug:
            w.writerow({k: r.get(k, "") for k in ["Configuration", "F1", "AUC", "Prec.", "Rec.", "ΔF1"]})

    out_json.write_text(json.dumps(rows_debug, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", out_csv)
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()

