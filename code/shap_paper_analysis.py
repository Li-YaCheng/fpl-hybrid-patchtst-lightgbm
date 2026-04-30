from __future__ import annotations

"""
Paper-aligned SHAP visualizations for Section 7 (Hybrid Stage-2 LightGBM).

Targets (matching the paper narrative):
- Fig5: Top-10 global feature importance (mean |SHAP|) for Stage-2 LightGBM on Z=[flat(X); mu_p; sigma_p]
- Fig6: SHAP beeswarm distribution for top features
- Fig7: Temporal importance (last 10 steps), mean |SHAP| (flat block), paper-style y ticks (0.005)

Inputs:
  - artifacts/npz/paper20d_L30.npz  (from sequence_generator_paper20d.py)

Outputs (figures/):
  - shap_paper_fig5_top10_bar.png
  - shap_paper_fig6_summary.png
  - shap_paper_fig7_temporal_last10.png
  - shap_paper_triptych.png
"""

import math
from pathlib import Path

import numpy as np

from patchtst_paper import PatchTSTPaperConfig, fit_predict_all


PKG_ROOT = Path(__file__).resolve().parents[1]
FAST = (PKG_ROOT / ".fast").exists()


def _load_npz():
    npz = PKG_ROOT / "artifacts" / "npz" / "paper20d_L30.npz"
    if not npz.exists():
        raise SystemExit(f"Missing: {npz}. Run data_loader_paper20d.py + sequence_generator_paper20d.py first.")
    z = np.load(npz, allow_pickle=True)
    X_tr = z["X_train"].astype(np.float32)
    y_tr = z["y_train"].astype(int)
    X_va = z["X_val"].astype(np.float32)
    y_va = z["y_val"].astype(int)
    X_te = z["X_test"].astype(np.float32)
    y_te = z["y_test"].astype(int)
    feat_names = z.get("feature_names", None)
    if feat_names is None:
        feat_names = np.asarray([f"f{i}" for i in range(X_tr.shape[-1])], dtype=object)
    feat_names = [str(x) for x in feat_names.tolist()]
    return X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names


def _train_lgbm(Ztr: np.ndarray, ytr: np.ndarray, Zva: np.ndarray, yva: np.ndarray):
    import lightgbm as lgb  # type: ignore

    dtr = lgb.Dataset(Ztr, label=ytr)
    dva = lgb.Dataset(Zva, label=yva, reference=dtr)
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
        "seed": 42,
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


def _feature_names_L30(feat_names_20: list[str]) -> list[str]:
    # Flatten order matches reshape: (L,d) -> [t0 features, t1 features, ...] if we flatten row-major.
    # Here X is stored as (N,L,D) where last index is feature.
    # In our code we use X.reshape(N, -1) which yields order: time-major then feature-major.
    # We label timesteps as t0=most recent (last step), t29=oldest.
    L = 30
    D = len(feat_names_20)
    names = []
    # Our sequence tensor order is chronological (older -> newer) in window; last index is t0.
    # After reshape, timestep index i corresponds to window[i]. We map i -> t-(L-i) so last is t0.
    for i in range(L):
        rel = i - (L - 1)  # last step => 0, first => -(L-1)
        for j in range(D):
            names.append(f"{feat_names_20[j]}_t{rel:+d}".replace("+", ""))
    # Stage-1 probability statistics (paper naming)
    names.append("stage1_mean_p")
    names.append("stage1_std_p")
    return names


def main() -> None:
    try:
        import shap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.ticker as mticker  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing SHAP/matplotlib. Import error: {e}")

    X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names = _load_npz()
    out_dir = PKG_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========= Style (match paper screenshots) =========
    # Colors sampled to match your provided reference images.
    C_PINK = "#f3b6cf"
    C_BLUE = "#cfe3f6"
    C_BLUE_EDGE = "#8db6d6"
    C_PURPLE = "#9b76c8"
    C_YELLOW = "#f2cc4d"
    GRID = "#e6e6e6"
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "font.family": "DejaVu Sans",
        }
    )

    # Stage-1 PatchTST ensemble -> mu/sigma on train/val/test
    seeds = [42, 43, 44]
    cfg = PatchTSTPaperConfig(max_epochs=10 if FAST else 60)
    p_tr_list, p_va_list, p_te_list = [], [], []
    for s in seeds:
        p_tr_s, _e_tr, p_va_s, _e_va, p_te_s, _e_te = fit_predict_all(
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

    Xtr_flat = X_tr.reshape((len(X_tr), -1))
    Xva_flat = X_va.reshape((len(X_va), -1))
    Xte_flat = X_te.reshape((len(X_te), -1))
    Ztr = np.concatenate([Xtr_flat, mu_tr.reshape(-1, 1), sig_tr.reshape(-1, 1)], axis=1)
    Zva = np.concatenate([Xva_flat, mu_va.reshape(-1, 1), sig_va.reshape(-1, 1)], axis=1)
    Zte = np.concatenate([Xte_flat, mu_te.reshape(-1, 1), sig_te.reshape(-1, 1)], axis=1)

    booster, best_iter = _train_lgbm(Ztr, y_tr, Zva, y_va)

    # ========= SHAP on TEST (paper: n=120) =========
    n_test = min(120, Zte.shape[0])
    Zs = Zte[:n_test]
    names = _feature_names_L30(feat_names)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(Zs)
    # SHAP may return list for binary classifiers (class0,class1). Use positive class if list.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    # Fig6: TreeSHAP summary (beeswarm)
    out6 = out_dir / "shap_paper_fig6_summary.png"
    plt.figure(figsize=(10.8, 6.0), dpi=220)
    shap.summary_plot(shap_values, Zs, feature_names=names, show=False, max_display=10)
    plt.title(f"TreeSHAP summary (Hybrid LightGBM on Z) | test (n={n_test})")
    plt.xlabel("SHAP value (impact on model output)")
    plt.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out6)
    plt.close()

    # Fig5: Top-10 features by mean |SHAP|
    out5 = out_dir / "shap_paper_fig5_top10_bar.png"
    sv = np.asarray(shap_values)
    mean_abs = np.mean(np.abs(sv), axis=0)
    topk = 10
    idx = np.argsort(mean_abs)[::-1][:topk]
    vals = mean_abs[idx][::-1]  # reverse for barh bottom->top
    labs = [names[i] for i in idx][::-1]
    colors = [C_BLUE] * topk
    edges = [C_BLUE_EDGE] * topk
    # highlight stage1_mean_p in pink (top feature in paper)
    for i, lab in enumerate(labs):
        if lab == "stage1_mean_p":
            colors[i] = C_PINK
            edges[i] = "#d48aa8"
    fig, ax = plt.subplots(figsize=(11.2, 4.3), dpi=220)
    ax.barh(range(topk), vals, color=colors, edgecolor=edges, linewidth=1.0)
    ax.set_yticks(range(topk))
    ax.set_yticklabels(labs)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Top-10 features by mean |SHAP| (Hybrid LightGBM)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    fig.savefig(out5)
    plt.close(fig)

    # Fig7: temporal importance last 10 steps
    out7 = out_dir / "shap_paper_fig7_temporal_last10.png"
    mean_abs = np.mean(np.abs(sv), axis=0)  # (602,)
    L, D = 30, len(feat_names)
    # exclude mu/sigma from timestep aggregation
    mean_abs_flat = mean_abs[: L * D].reshape(L, D)
    # Paper Fig7: mean |SHAP| (flat block) = mean over D coords after mean |SHAP| over test rows.
    step_imp = mean_abs_flat.mean(axis=1)  # (L,)
    # last 10 timesteps (t-9..t0)
    last10 = step_imp[-10:]
    xs = [f"t-{i}" for i in range(9, 0, -1)] + ["t-0"]
    fig7, ax7 = plt.subplots(figsize=(11.2, 3.5), dpi=220)
    bar_colors = [C_PURPLE] * 10
    # highlight t-1 and t-0 in yellow as in the paper screenshot
    bar_colors[-2] = C_YELLOW
    bar_colors[-1] = C_YELLOW
    ax7.bar(range(10), last10, color=bar_colors, edgecolor="#666666", linewidth=0.8)
    ax7.set_xticks(range(10))
    ax7.set_xticklabels(xs)
    ax7.set_ylabel("mean |SHAP| (flat block)")
    ymax = float(np.max(last10)) if len(last10) else 0.0
    ax7.minorticks_off()
    # Paper-style y-axis: 0.005 tick step when values stay in ~[0, 0.03] (avoid tick explosion).
    if ymax > 0 and ymax <= 0.045:
        yhi = float(np.ceil(max(0.03, ymax * 1.08) / 0.005) * 0.005)
        ax7.set_ylim(0.0, yhi)
        ax7.yaxis.set_major_locator(mticker.MultipleLocator(0.005))
        yfmt = mticker.FormatStrFormatter("%.3f")
    else:
        yhi = ymax * 1.12 if ymax > 0 else 1e-6
        ax7.set_ylim(0.0, yhi)
        ax7.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="lower"))
        yfmt = mticker.ScalarFormatter(useOffset=False)
        yfmt.set_scientific(False)
    ax7.yaxis.set_major_formatter(yfmt)
    ax7.set_title("Temporal importance (last 10 steps) | Hybrid LightGBM")
    ax7.grid(False)
    for spine in ax7.spines.values():
        spine.set_visible(True)
    fig7.tight_layout()
    fig7.savefig(out7)
    plt.close(fig7)

    # Compose triptych (a)(b)(c)
    from PIL import Image, ImageDraw, ImageFont

    def _try_font(size: int):
        for p in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
            try:
                if Path(p).exists():
                    return ImageFont.truetype(p, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    ia = Image.open(out5).convert("RGB")
    ib = Image.open(out6).convert("RGB")
    ic = Image.open(out7).convert("RGB")
    H = max(ia.size[1], ib.size[1], ic.size[1])

    def norm_h(im: Image.Image) -> Image.Image:
        if im.size[1] == H:
            return im
        return im.resize((int(im.size[0] * H / im.size[1]), H), Image.Resampling.LANCZOS)

    ia, ib, ic = norm_h(ia), norm_h(ib), norm_h(ic)
    gap = 18
    W = ia.size[0] + ib.size[0] + ic.size[0] + gap * 2
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    x = 0
    canvas.paste(ia, (x, 0))
    x += ia.size[0] + gap
    canvas.paste(ib, (x, 0))
    x += ib.size[0] + gap
    canvas.paste(ic, (x, 0))

    d = ImageDraw.Draw(canvas)
    font = _try_font(28)
    d.text((16, 10), "(a)", fill=(0, 0, 0), font=font)
    d.text((ia.size[0] + gap + 16, 10), "(b)", fill=(0, 0, 0), font=font)
    d.text((ia.size[0] + gap + ib.size[0] + gap + 16, 10), "(c)", fill=(0, 0, 0), font=font)

    out_trip = out_dir / "shap_paper_triptych.png"
    canvas.save(out_trip, format="PNG", optimize=True)

    print("Wrote:", out5)
    print("Wrote:", out6)
    print("Wrote:", out7)
    print("Wrote:", out_trip)


if __name__ == "__main__":
    main()

