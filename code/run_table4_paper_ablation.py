from __future__ import annotations

"""
Paper-aligned Table 4: Ablation Study Results.

Objects (exactly as the paper):
1 L=30, Hybrid (PatchTST p, seed 42)
2 L=30, Hybrid (seed 123)
3 L=30, Hybrid (seed 456)
4 L=30, LightGBM (flat, no PatchTST)
5 L=15, Hybrid (PatchTST emb, short)
6 L=15, LightGBM (flat, no temporal)

Protocol:
- Train/Val/Test splits are season-based from paper20d_L30.npz
- Threshold τ* selected on Val only (grid [0.03,0.97], step 0.005)
- ΔF1 relative to row 1

Implementation details (paper-driven):
- Stage-1 PatchTST-lite uses patch_len=5, d_model=64, heads=4, layers=2, focal(α=0.75,γ=1.8), dropout=0.5, l2=1e-4
- Probabilistic features: multi-seed ensemble K=3 -> μp, σp
- Stage-2 LightGBM fusion uses Z=[flat(X); μp; σp] (602-D for L=30)
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


def _load_npz_L30() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _hybrid_prob_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    *,
    base_seed: int,
    cfg: PatchTSTPaperConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train K=3 PatchTST models with seeds {base_seed, base_seed+1, base_seed+2}
    and return μ,σ for train/val/test.
    """
    seeds = [int(base_seed), int(base_seed) + 1, int(base_seed) + 2]
    ptr, pva, pte = [], [], []
    for s in seeds:
        p_tr_s, _emb_tr, p_va_s, _emb_va, p_te_s, _emb_te = fit_predict_all(
            X_tr, y_tr, X_va, y_va, X_te, seed=int(s), cfg=cfg
        )
        ptr.append(p_tr_s)
        pva.append(p_va_s)
        pte.append(p_te_s)
    Ptr = np.stack(ptr, axis=0)
    Pva = np.stack(pva, axis=0)
    Pte = np.stack(pte, axis=0)
    return (
        Ptr.mean(axis=0),
        Ptr.std(axis=0, ddof=0),
        Pva.mean(axis=0),
        Pva.std(axis=0, ddof=0),
        Pte.mean(axis=0),
        Pte.std(axis=0, ddof=0),
    )


def main() -> None:
    X_tr, y_tr, X_va, y_va, X_te, y_te = _load_npz_L30()

    out_dir = PKG_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "table4_ablation_study.csv"
    out_json = out_dir / "table4_ablation_study.json"

    Xtr_flat = X_tr.reshape((len(X_tr), -1))
    Xva_flat = X_va.reshape((len(X_va), -1))
    Xte_flat = X_te.reshape((len(X_te), -1))

    rows: list[dict] = []

    # Exp.1-3: L=30 Hybrid, varying seeds
    for seed in ([42] if FAST else [42, 123, 456]):
        t0 = time.time()
        cfg = PatchTSTPaperConfig(max_epochs=10 if FAST else 60)
        mu_tr, sig_tr, mu_va, sig_va, mu_te, sig_te = _hybrid_prob_features(
            X_tr, y_tr, X_va, y_va, X_te, base_seed=int(seed), cfg=cfg
        )
        Ztr = np.concatenate([Xtr_flat, mu_tr.reshape(-1, 1), sig_tr.reshape(-1, 1)], axis=1)
        Zva = np.concatenate([Xva_flat, mu_va.reshape(-1, 1), sig_va.reshape(-1, 1)], axis=1)
        Zte = np.concatenate([Xte_flat, mu_te.reshape(-1, 1), sig_te.reshape(-1, 1)], axis=1)

        booster, bi = _train_lgbm(Ztr, y_tr, Zva, y_va, seed=int(seed))
        pva = _predict_lgbm(booster, Zva, bi)
        thr = best_thr_by_val_f1(y_va, pva)
        pte = _predict_lgbm(booster, Zte, bi)
        m = metrics_at_thr(y_te, pte, thr)
        if int(seed) == 42:
            cfg_name = "L=30, Hybrid (PatchTST p, seed 42)"
        else:
            cfg_name = f"L=30, Hybrid (seed {seed})"
        rows.append(
            {
                "Configuration": cfg_name,
                **m,
                "ΔF1": "",
                "_thr": float(thr),
                "_best_iter": int(bi),
                "_train_time_s": float(time.time() - t0),
            }
        )

    base_f1 = float(rows[0]["F1"])
    for i, r in enumerate(rows):
        if i == 0:
            r["ΔF1"] = "—"
        else:
            r["ΔF1"] = f"{(float(r['F1']) - base_f1):+.2%}"

    # Exp.4: L=30 LightGBM (flat, no PatchTST)
    t0 = time.time()
    booster4, bi4 = _train_lgbm(Xtr_flat, y_tr, Xva_flat, y_va, seed=42)
    pva4 = _predict_lgbm(booster4, Xva_flat, bi4)
    thr4 = best_thr_by_val_f1(y_va, pva4)
    pte4 = _predict_lgbm(booster4, Xte_flat, bi4)
    m4 = metrics_at_thr(y_te, pte4, thr4)
    rows.append(
        {
            "Configuration": "L=30, LightGBM (flat, no PatchTST)",
            **m4,
            "ΔF1": f"{(float(m4['F1']) - base_f1):+.2%}",
            "_thr": float(thr4),
            "_best_iter": int(bi4),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # For L=15, derive from L=30 tensor by taking last 15 steps (paper ablation).
    X_tr15, X_va15, X_te15 = X_tr[:, -15:, :], X_va[:, -15:, :], X_te[:, -15:, :]
    Xtr15_flat = X_tr15.reshape((len(X_tr15), -1))
    Xva15_flat = X_va15.reshape((len(X_va15), -1))
    Xte15_flat = X_te15.reshape((len(X_te15), -1))

    # Exp.5: L=15 Hybrid (PatchTST emb, short)
    t0 = time.time()
    # "emb, short": a smaller temporal encoder; keep patch_len=5 to match divisibility.
    cfg15 = PatchTSTPaperConfig(seq_len=15, patch_len=5, d_model=32, n_heads=4, n_layers=1, ff_dim=96, max_epochs=10 if FAST else 40)
    # single seed for this ablation row as in paper table wording
    p_tr5, emb_tr5, p_va5, emb_va5, p_te5, emb_te5 = fit_predict_all(X_tr15, y_tr, X_va15, y_va, X_te15, seed=42, cfg=cfg15)
    Ztr5 = np.concatenate([Xtr15_flat, emb_tr5.astype(np.float32)], axis=1)
    Zva5 = np.concatenate([Xva15_flat, emb_va5.astype(np.float32)], axis=1)
    Zte5 = np.concatenate([Xte15_flat, emb_te5.astype(np.float32)], axis=1)
    booster5, bi5 = _train_lgbm(Ztr5, y_tr, Zva5, y_va, seed=42)
    pva5b = _predict_lgbm(booster5, Zva5, bi5)
    thr5 = best_thr_by_val_f1(y_va, pva5b)
    pte5b = _predict_lgbm(booster5, Zte5, bi5)
    m5 = metrics_at_thr(y_te, pte5b, thr5)
    rows.append(
        {
            "Configuration": "L=15, Hybrid (PatchTST emb, short)",
            **m5,
            "ΔF1": f"{(float(m5['F1']) - base_f1):+.2%}",
            "_thr": float(thr5),
            "_best_iter": int(bi5),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # Exp.6: L=15 LightGBM (flat, no temporal) -> interpret as last-step only (t0).
    t0 = time.time()
    Xtr_nt = X_tr15[:, -1, :].reshape((len(X_tr15), -1))
    Xva_nt = X_va15[:, -1, :].reshape((len(X_va15), -1))
    Xte_nt = X_te15[:, -1, :].reshape((len(X_te15), -1))
    booster6, bi6 = _train_lgbm(Xtr_nt, y_tr, Xva_nt, y_va, seed=42)
    pva6 = _predict_lgbm(booster6, Xva_nt, bi6)
    thr6 = best_thr_by_val_f1(y_va, pva6)
    pte6 = _predict_lgbm(booster6, Xte_nt, bi6)
    m6 = metrics_at_thr(y_te, pte6, thr6)
    rows.append(
        {
            "Configuration": "L=15, LightGBM (flat, no temporal)",
            **m6,
            "ΔF1": f"{(float(m6['F1']) - base_f1):+.2%}",
            "_thr": float(thr6),
            "_best_iter": int(bi6),
            "_train_time_s": float(time.time() - t0),
        }
    )

    # Write CSV (paper columns only)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Configuration", "F1", "AUC", "Prec.", "Rec.", "ΔF1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["Configuration", "F1", "AUC", "Prec.", "Rec.", "ΔF1"]})

    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", out_csv)
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()

