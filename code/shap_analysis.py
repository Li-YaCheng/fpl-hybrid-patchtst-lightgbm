from __future__ import annotations

"""
SHAP analysis for the reconstructed holdout predictor.

Inputs:
  - artifacts/models/lgbm_flat_L30.txt
  - artifacts/npz/processed_data_v2_L30_padpred.npz

Outputs (figures/):
  - shap_lgbm_L30_summary.png
  - shap_lgbm_L30_bar.png

Notes:
- The original project produced richer SHAP visuals (including temporal grouping). This script
  provides the core SHAP summary/bar plots for the flattened LightGBM model.
"""

from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]
FAST = (PKG_ROOT / ".fast").exists()


def main() -> None:
    try:
        import lightgbm as lgb  # type: ignore
        import shap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing SHAP dependencies. Use upload/environment.yml (numpy<2).\n"
            f"Import error: {e}"
        )

    model_path = PKG_ROOT / "artifacts" / "models" / "lgbm_flat_L30.txt"
    npz_path = PKG_ROOT / "artifacts" / "npz" / "processed_data_v2_L30_padpred.npz"
    out_dir = PKG_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Missing: {model_path}. Run run_predict_holdout_25_26.py first.")
    if not npz_path.exists():
        raise SystemExit(f"Missing: {npz_path}. Run sequence_generator_15dim.py first.")

    z = np.load(npz_path, allow_pickle=True)
    X = z["X_train"].astype(np.float32)
    y = z["y_train"].astype(int)
    feat_names = z.get("feature_names", None)
    if feat_names is None:
        feat_names = np.asarray([f"f{i}" for i in range(X.shape[-1])], dtype=object)
    feat_names = [str(x) for x in feat_names.tolist()]

    # Flatten features with temporal index
    L, D = X.shape[1], X.shape[2]
    Xf = X.reshape((X.shape[0], -1))
    names = [f"{feat_names[j]}@t{-L+1+i}" for i in range(L) for j in range(D)]

    # sample for speed
    n = min(800 if FAST else 6000, Xf.shape[0])
    Xs = Xf[:n]

    booster = lgb.Booster(model_file=str(model_path))
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(Xs)

    # summary (beeswarm)
    plt.figure(figsize=(9, 6), dpi=200)
    shap.summary_plot(shap_values, Xs, feature_names=names, show=False, max_display=25)
    plt.tight_layout()
    out1 = out_dir / "shap_lgbm_L30_summary.png"
    plt.savefig(out1)
    plt.close()

    # bar (mean |SHAP|)
    plt.figure(figsize=(9, 5), dpi=200)
    shap.summary_plot(shap_values, Xs, feature_names=names, plot_type="bar", show=False, max_display=25)
    plt.tight_layout()
    out2 = out_dir / "shap_lgbm_L30_bar.png"
    plt.savefig(out2)
    plt.close()

    print("Wrote:", out1)
    print("Wrote:", out2)


if __name__ == "__main__":
    main()

