from __future__ import annotations

"""
Table-4 style ablation study (reconstructed to match paper objects).

This script is meant to output rows that match the paper's Table 4 objects:
  1) L=30, Hybrid (PatchTST P, seed 42)
  2) L=30, Hybrid (seed 123)
  3) L=30, Hybrid (seed 456)
  4) L=30, LightGBM (flat, no PatchTST)
  5) L=15, Hybrid (PatchTST emb, short)
  6) L=15, LightGBM (flat, no temporal)

Important:
- The original PatchTST/Hybrid code was not available in this repo snapshot.
  We implement a lightweight temporal encoder (`patchtst_light.py`) to produce:
    - p: probability feature (PatchTST P)
    - emb: embedding vector (PatchTST emb)
- Evaluation protocol: threshold is selected on val by F1, then metrics are reported on test.

Inputs:
  - artifacts/npz/processed_data_v2_L30_padpred.npz

Outputs:
  - tables/table4_ablation_study.csv
  - tables/table4_ablation_study.json
"""

import csv
import time
from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]

FAST = (Path(__file__).resolve().parents[1] / ".fast").exists()
SEEDS = [42, 123, 456]


def _best_thr_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score  # type: ignore

    best_thr, best = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (p >= thr).astype(int))
        if f1 > best:
            best = f1
            best_thr = float(thr)
    return best_thr


def _eval_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # type: ignore

    y_pred = (p >= thr).astype(int)
    return {
        "F1": float(f1_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, p)),
        "Prec.": float(precision_score(y_true, y_pred, zero_division=0)),
        "Rec.": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    try:
        import lightgbm as lgb  # type: ignore
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

    out_dir = PKG_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # L=15 views are derived from the L=30 tensor by taking the last 15 steps.
    X_tr_15 = X_tr[:, -15:, :]
    X_va_15 = X_va[:, -15:, :]
    X_te_15 = X_te[:, -15:, :]

    def train_lgbm_prob(
        Xtr_flat: np.ndarray, Xva_flat: np.ndarray, Xte_flat: np.ndarray, *, seed: int
    ) -> tuple[np.ndarray, float, dict]:
        dtr = lgb.Dataset(Xtr_flat, label=y_tr)
        dva = lgb.Dataset(Xva_flat, label=y_va, reference=dtr)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": int(seed),
            "feature_pre_filter": False,
        }
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=300 if FAST else 3000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(20 if FAST else 80, verbose=False)],
        )
        best_iter = int(getattr(booster, "best_iteration", 0) or 0)
        pva = booster.predict(Xva_flat, num_iteration=best_iter or None)
        thr = _best_thr_f1(y_va, pva)
        pte = booster.predict(Xte_flat, num_iteration=best_iter or None)
        m = _eval_metrics(y_te, pte, thr)
        m["_thr"] = float(thr)
        m["_best_iter"] = int(best_iter)
        return pte, float(thr), m

    # === Config 1-3: L=30, Hybrid (PatchTST P, different seeds) ===
    # stage-1: PatchTST -> probability p
    # stage-2: LightGBM on [flattened seq, p] -> final prediction
    from patchtst_light import PatchTSTConfig, fit_predict_embed

    def patch_cfg_full(L: int) -> PatchTSTConfig:
        if FAST:
            return PatchTSTConfig(seq_len=L, d_model=32, n_heads=4, n_layers=1, max_epochs=6, patience=2, batch_size=256)
        return PatchTSTConfig(seq_len=L, d_model=64, n_heads=4, n_layers=2, max_epochs=25, patience=5, batch_size=256)

    def patch_cfg_short(L: int) -> PatchTSTConfig:
        if FAST:
            return PatchTSTConfig(seq_len=L, d_model=16, n_heads=2, n_layers=1, max_epochs=6, patience=2, batch_size=256)
        return PatchTSTConfig(seq_len=L, d_model=32, n_heads=4, n_layers=1, max_epochs=20, patience=5, batch_size=256)

    def hybrid_patchP_L30(seed: int) -> dict:
        t0 = time.time()
        p_va, _emb_va, p_te, _emb_te = fit_predict_embed(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            seed=int(seed),
            cfg=patch_cfg_full(30),
        )
        _pva2, _emb_va2, p_tr, _emb_tr = fit_predict_embed(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_tr,
            seed=int(seed),
            cfg=patch_cfg_full(30),
        )
        Xtr = np.concatenate([X_tr.reshape((len(X_tr), -1)), np.zeros((len(X_tr), 1), dtype=np.float32)], axis=1)
        Xva = np.concatenate([X_va.reshape((len(X_va), -1)), p_va.reshape(-1, 1).astype(np.float32)], axis=1)
        Xte = np.concatenate([X_te.reshape((len(X_te), -1)), p_te.reshape(-1, 1).astype(np.float32)], axis=1)
        Xtr[:, -1] = p_tr.reshape(-1).astype(np.float32)

        _pte, thr, m = train_lgbm_prob(Xtr, Xva, Xte, seed=int(seed))
        if int(seed) == 42:
            cfg_name = f"L=30, Hybrid (PatchTST P, seed {seed})"
        else:
            # Match the paper row naming (seed-only for the other seeds)
            cfg_name = f"L=30, Hybrid (seed {seed})"
        out = {
            "Configuration": cfg_name,
            "F1": m["F1"],
            "AUC": m["AUC"],
            "Prec.": m["Prec."],
            "Rec.": m["Rec."],
            "_thr": m["_thr"],
            "_best_iter": m["_best_iter"],
            "_train_time_s": float(time.time() - t0),
        }
        return out

    # === Config 4: L=30, LightGBM (flat, no PatchTST) ===
    def lgbm_flat_L30(seed: int = 42) -> dict:
        t0 = time.time()
        Xtr = X_tr.reshape((len(X_tr), -1))
        Xva = X_va.reshape((len(X_va), -1))
        Xte = X_te.reshape((len(X_te), -1))
        _pte, _thr, m = train_lgbm_prob(Xtr, Xva, Xte, seed=int(seed))
        return {
            "Configuration": "L=30, LightGBM (flat, no PatchTST)",
            "F1": m["F1"],
            "AUC": m["AUC"],
            "Prec.": m["Prec."],
            "Rec.": m["Rec."],
            "_thr": m["_thr"],
            "_best_iter": m["_best_iter"],
            "_train_time_s": float(time.time() - t0),
        }

    # === Config 5: L=15, Hybrid (PatchTST emb, short) ===
    def hybrid_embshort_L15(seed: int = 42) -> dict:
        t0 = time.time()
        p_va, emb_va, p_te, emb_te = fit_predict_embed(
            X_tr_15,
            y_tr,
            X_va_15,
            y_va,
            X_te_15,
            seed=int(seed),
            cfg=patch_cfg_short(15),
        )
        # Train needs embedding too
        _pva2, _emb_va2, p_tr, emb_tr = fit_predict_embed(
            X_tr_15,
            y_tr,
            X_va_15,
            y_va,
            X_tr_15,
            seed=int(seed),
            cfg=patch_cfg_short(15),
        )
        Xtr = np.concatenate([X_tr_15.reshape((len(X_tr_15), -1)), emb_tr.astype(np.float32)], axis=1)
        Xva = np.concatenate([X_va_15.reshape((len(X_va_15), -1)), emb_va.astype(np.float32)], axis=1)
        Xte = np.concatenate([X_te_15.reshape((len(X_te_15), -1)), emb_te.astype(np.float32)], axis=1)

        _pte, _thr, m = train_lgbm_prob(Xtr, Xva, Xte, seed=int(seed))
        return {
            "Configuration": "L=15, Hybrid (PatchTST emb, short)",
            "F1": m["F1"],
            "AUC": m["AUC"],
            "Prec.": m["Prec."],
            "Rec.": m["Rec."],
            "_thr": m["_thr"],
            "_best_iter": m["_best_iter"],
            "_train_time_s": float(time.time() - t0),
        }

    # === Config 6: L=15, LightGBM (flat, no temporal) ===
    # Interpret "no temporal" as last-step only (single timestep features).
    def lgbm_laststep_L15(seed: int = 42) -> dict:
        t0 = time.time()
        Xtr = X_tr_15[:, -1, :].reshape((len(X_tr_15), -1))
        Xva = X_va_15[:, -1, :].reshape((len(X_va_15), -1))
        Xte = X_te_15[:, -1, :].reshape((len(X_te_15), -1))
        _pte, _thr, m = train_lgbm_prob(Xtr, Xva, Xte, seed=int(seed))
        return {
            "Configuration": "L=15, LightGBM (flat, no temporal)",
            "F1": m["F1"],
            "AUC": m["AUC"],
            "Prec.": m["Prec."],
            "Rec.": m["Rec."],
            "_thr": m["_thr"],
            "_best_iter": m["_best_iter"],
            "_train_time_s": float(time.time() - t0),
        }

    rows: list[dict] = []

    # 1-3: L=30 Hybrid rows (seed 42/123/456)
    hybrid_seeds = [42] if FAST else SEEDS
    for s in hybrid_seeds:
        rows.append(hybrid_patchP_L30(int(s)))

    # 4-6: fixed configs (seed 42 like the paper table)
    rows.append(lgbm_flat_L30(seed=42))
    rows.append(hybrid_embshort_L15(seed=42))
    rows.append(lgbm_laststep_L15(seed=42))

    # ΔF1 vs full model (row 1: L=30 Hybrid seed 42)
    base_f1 = float(rows[0]["F1"])
    for r in rows:
        if r is rows[0]:
            r["ΔF1"] = "—"
        else:
            r["ΔF1"] = f"{(float(r['F1']) - base_f1):+.2%}"

    # write CSV matching paper columns
    out_csv = out_dir / "table4_ablation_study.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Configuration", "F1", "AUC", "Prec.", "Rec.", "ΔF1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["Configuration", "F1", "AUC", "Prec.", "Rec.", "ΔF1"]})
    print("Wrote:", out_csv)

    # also write a debug json with extra fields
    import json

    out_json = out_dir / "table4_ablation_study.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()

