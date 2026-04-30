"""
Final fair comparison (reconstructed).

In the original work, this script compared the proposed method (15-dim + focal) against baselines
using the same generated sequence tensors and best-found configs.

Because the original training code is not recoverable in this snapshot, this script currently:
- reads `tables/baseline_comparison_results.csv`
- reads the shipped paper-ready predictions CSV (optional)
- writes a consolidated `tables/final_comparison_results.csv` that is stable and reproducible

You can extend this to include your reconstructed hybrid model once training scripts are added
(see run_hybrid_prob_cached.py and boost_15focal_ours.py in later steps).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[1]
FAST = (PKG_ROOT / ".fast").exists()


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [{k: v for k, v in row.items()} for row in r]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Fieldnames must cover all keys across rows.
    # Use a stable order: common metrics first, then the remaining keys alphabetically.
    preferred = [
        "model",
        "group",
        "f1",
        "auc",
        "accuracy",
        "precision",
        "recall",
        "threshold",
        "train_time_s",
        "best_iter",
        "source",
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = [k for k in preferred if k in keys] + sorted([k for k in keys if k not in preferred])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # fill missing keys with empty string for CSV consistency
            rr = {k: r.get(k, "") for k in fieldnames}
            w.writerow(rr)


def _best_thr_f1(y_true, p) -> float:
    from sklearn.metrics import f1_score  # type: ignore

    best_thr, best = 0.5, -1.0
    for thr in [x / 100 for x in range(5, 96)]:
        f1 = f1_score(y_true, (p >= thr).astype(int))
        if f1 > best:
            best = f1
            best_thr = float(thr)
    return best_thr


def _eval(y_true, p, thr: float) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score  # type: ignore

    y_pred = (p >= thr).astype(int)
    out = {
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    return out


def _train_eval_ours(sweep_best: Path) -> dict:
    """
    Train LightGBM(flat seq) on X_train and evaluate on X_test.
    Threshold is selected on X_val (F1).
    """
    try:
        import numpy as np  # type: ignore
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing ML deps. Create the clean env from upload/environment.yml (numpy<2).\n"
            f"Import error: {e}"
        )

    npz_path = PKG_ROOT / "artifacts" / "npz" / "processed_data_v2_L30_padpred.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing: {npz_path}. Run sequence_generator_15dim.py first.")

    z = np.load(npz_path, allow_pickle=True)
    X_tr, y_tr = z["X_train"].astype("float32"), z["y_train"].astype(int)
    X_va, y_va = z["X_val"].astype("float32"), z["y_val"].astype(int)
    X_te, y_te = z["X_test"].astype("float32"), z["y_test"].astype(int)

    Xtr = X_tr.reshape((len(X_tr), -1))
    Xva = X_va.reshape((len(X_va), -1))
    Xte = X_te.reshape((len(X_te), -1))

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 0.0,
    }

    best_iter = 0
    if sweep_best.exists():
        cfg = json.loads(sweep_best.read_text(encoding="utf-8"))
        # Map known keys from sweep output
        params["learning_rate"] = float(cfg.get("learning_rate", params["learning_rate"]))
        params["num_leaves"] = int(cfg.get("num_leaves", params["num_leaves"]))
        params["min_data_in_leaf"] = int(cfg.get("min_data_in_leaf", params["min_data_in_leaf"]))
        params["feature_fraction"] = float(cfg.get("feature_fraction", params["feature_fraction"]))
        params["bagging_fraction"] = float(cfg.get("bagging_fraction", params["bagging_fraction"]))
        params["lambda_l2"] = float(cfg.get("lambda_l2", params["lambda_l2"]))
        best_iter = int(cfg.get("best_iter", 0) or 0)

    dtr = lgb.Dataset(Xtr, label=y_tr)
    dva = lgb.Dataset(Xva, label=y_va, reference=dtr)

    t0 = time.time()
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=max(200 if FAST else 600, best_iter or 0),
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(20 if FAST else 80, verbose=False)],
    )
    train_time = float(time.time() - t0)
    bi = int(getattr(booster, "best_iteration", 0) or 0)
    if bi > 0:
        best_iter = bi

    pva = booster.predict(Xva, num_iteration=best_iter or None)
    thr = _best_thr_f1(y_va, pva)
    pte = booster.predict(Xte, num_iteration=best_iter or None)
    m = _eval(y_te, pte, thr)
    m["threshold"] = float(thr)
    m["train_time_s"] = train_time
    m["best_iter"] = int(best_iter)
    return m


def main() -> None:
    base = PKG_ROOT / "tables"
    base.mkdir(parents=True, exist_ok=True)
    baseline_csv = base / "baseline_comparison_results.csv"
    sweep_best = base / "lgbm_flat_L30_sweep_best.json"
    out_csv = base / "final_comparison_results.csv"
    out_table5 = base / "table5_cross_model_comparison.csv"

    rows: list[dict] = []
    if baseline_csv.exists():
        for r in _read_csv(baseline_csv):
            r2 = dict(r)
            r2["group"] = "baseline"
            rows.append(r2)

    # "Ours" (reconstructed): train/eval once and fill metrics.
    ours_metrics = _train_eval_ours(sweep_best)
    ours = {
        "model": "Ours (reconstructed LGBM-flat L30)",
        "f1": ours_metrics["f1"],
        "auc": ours_metrics["auc"],
        "accuracy": ours_metrics["accuracy"],
        "precision": ours_metrics["precision"],
        "recall": ours_metrics["recall"],
        "threshold": ours_metrics["threshold"],
        "train_time_s": ours_metrics["train_time_s"],
        "best_iter": ours_metrics["best_iter"],
        "group": "ours",
    }
    if sweep_best.exists():
        ours["source"] = "lgbm_flat_L30_sweep_best.json"
    else:
        ours["source"] = "default_params"
    rows.append(ours)

    _write_csv(out_csv, rows)
    print("Wrote:", out_csv)

    # Write a Table-5 style view matching the paper layout/rows.
    # Rows are fixed to: Ours, LightGBM, LightGBM (flat L=30), XGBoost, Random Forest, PatchTST (end-to-end)
    # If a row is not available in the reconstructed pipeline, keep metrics blank (do NOT substitute other models).
    def pick(model_name: str) -> dict | None:
        for r in rows:
            if r.get("model") == model_name:
                return r
        return None

    # Map from our internal baseline names to paper row names
    mapping = {
        "LightGBM": "LightGBM(last-step)",
        "LightGBM (flat L=30)": "Ours (reconstructed LGBM-flat L30)",
        "XGBoost": "XGBoost(last-step)",
        "Random Forest": "RF(last-step)",
    }

    # Ours(Hybrid) and PatchTST(end-to-end) are not reconstructed yet; leave blank.
    ours_row_name = "Ours (L=30, Hybrid)"
    patchtst_row_name = "PatchTST (end-to-end)"

    # Compute ΔF1 vs Ours (paper row)
    ours_f1 = float(ours.get("f1") or 0.0)

    def fmt_num(v) -> str:
        try:
            x = float(v)
        except Exception:
            return ""
        if x != x:
            return ""
        return f"{x:.4f}"

    def row_from(src: dict | None, cfg: str, *, df1_from: float | None) -> dict:
        f1 = None if src is None else src.get("f1")
        f1v = None
        try:
            f1v = float(f1) if f1 is not None and f1 != "" else None
        except Exception:
            f1v = None
        df1 = ""
        if df1_from is not None and f1v is not None:
            df1 = f"{(f1v - df1_from):+.2%}"
        return {
            "Configuration": cfg,
            "F1": fmt_num("" if src is None else src.get("f1", "")),
            "AUC": fmt_num("" if src is None else src.get("auc", "")),
            "Prec.": fmt_num("" if src is None else src.get("precision", "")),
            "Rec.": fmt_num("" if src is None else src.get("recall", "")),
            "ΔF1": df1 if cfg != ours_row_name else "—",
        }

    t5_rows = [
        row_from(None, ours_row_name, df1_from=None),
        row_from(pick(mapping["LightGBM"]), "LightGBM", df1_from=ours_f1),
        row_from(pick(mapping["LightGBM (flat L=30)"]), "LightGBM (flat L=30)", df1_from=ours_f1),
        row_from(pick(mapping["XGBoost"]), "XGBoost", df1_from=ours_f1),
        row_from(pick(mapping["Random Forest"]), "Random Forest", df1_from=ours_f1),
        row_from(None, patchtst_row_name, df1_from=ours_f1),
    ]

    out_table5.parent.mkdir(parents=True, exist_ok=True)
    with out_table5.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Configuration", "F1", "AUC", "Prec.", "Rec.", "ΔF1"])
        w.writeheader()
        w.writerows(t5_rows)
    print("Wrote:", out_table5)


if __name__ == "__main__":
    main()

