from __future__ import annotations

"""
Predict on 2025–26 holdout sequences (reconstructed).

This script produces a paper-ready CSV compatible with:
- code/render_paper_figures_png.py
- code/make_holdout_25_26_compare_svg.py

Inputs:
  - artifacts/npz/processed_data_v2_L30_padpred.npz
  - data/processed/predict_25_26_sequences_meta.csv

Outputs:
  - artifacts/models/lgbm_flat_L30.txt
  - artifacts/pred/predict_25_26_hybrid_predictions_lgbm_L30.csv
  - data/predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv (for figure scripts; kept for compatibility)

Notes:
- This reconstruction uses LightGBM on flattened sequences as a strong, runnable baseline.
- The original project used a hybrid (deep stage-1 + LGBM stage-2). You can later swap the model
  while keeping the output CSV schema identical.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np


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


def main() -> None:
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        raise SystemExit(
            "LightGBM not available. Use upload/environment.yml (numpy<2).\n"
            f"Import error: {e}"
        )

    npz_path = PKG_ROOT / "artifacts" / "npz" / "processed_data_v2_L30_padpred.npz"
    meta_path = PKG_ROOT / "data" / "processed" / "predict_25_26_sequences_meta.csv"
    if not npz_path.exists():
        raise SystemExit(f"Missing: {npz_path}. Run sequence_generator_15dim.py first.")
    if not meta_path.exists():
        raise SystemExit(f"Missing: {meta_path}. Run sequence_generator_15dim.py first.")

    z = np.load(npz_path, allow_pickle=True)
    X_tr, y_tr = z["X_train"].astype(np.float32), z["y_train"].astype(int)
    X_va, y_va = z["X_val"].astype(np.float32), z["y_val"].astype(int)
    X_te, y_te = z["X_test"].astype(np.float32), z["y_test"].astype(int)
    X_pred = z["X_predict"].astype(np.float32)
    y_pred = z["y_predict"].astype(int)

    meta = _read_meta(meta_path)
    if len(meta) != len(X_pred):
        # best effort: keep aligned if possible
        raise SystemExit(f"Meta rows ({len(meta)}) != X_predict ({len(X_pred)}).")

    # Train on train+val+test (deployment-style) and predict holdout
    X_all = np.concatenate([X_tr, X_va, X_te], axis=0).reshape((len(y_tr) + len(y_va) + len(y_te), -1))
    y_all = np.concatenate([y_tr, y_va, y_te], axis=0)

    dtr = lgb.Dataset(X_all, label=y_all)
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
        "seed": 42,
    }
    booster = lgb.train(params, dtr, num_boost_round=200 if FAST else 1200)

    out_model = PKG_ROOT / "artifacts" / "models" / "lgbm_flat_L30.txt"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_model))

    p_pred = booster.predict(X_pred.reshape((len(X_pred), -1)))

    # Write paper-ready CSV schema
    out_pred = PKG_ROOT / "artifacts" / "pred" / "predict_25_26_hybrid_predictions_lgbm_L30.csv"
    out_pred.parent.mkdir(parents=True, exist_ok=True)

    # Also write to the legacy filename expected by some downstream scripts/figures
    out_legacy = PKG_ROOT / "data" / "predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv"

    header = [
        "player_name",
        "season_x",
        "round",
        "kickoff_time",
        "total_points",
        "y_true",
        "hybrid_proba",
    ]

    def write(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for m, p, yy in zip(meta, p_pred, y_pred):
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

    print("Wrote model:", out_model)
    print("Wrote:", out_pred)
    print("Wrote:", out_legacy)


if __name__ == "__main__":
    main()

