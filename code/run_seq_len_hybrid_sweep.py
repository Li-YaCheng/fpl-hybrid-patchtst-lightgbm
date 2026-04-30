from __future__ import annotations

"""
Seq-len sweep (reconstructed) — find best sequence length.

Reads the flat 15-dim dataset and, for each candidate L, builds sequences and evaluates
a fast model (LightGBM on flattened sequences) with threshold tuned on validation F1.

Outputs:
  - tables/seq_len_results_15dim.csv

This matches the *purpose* of the original sweep (select L=30), even though the exact
model architecture from the deleted scripts cannot be recovered here.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]


def _load_feature_names() -> list[str]:
    meta = PKG_ROOT / "data" / "processed" / "flat_15dim_meta.json"
    if not meta.exists():
        raise SystemExit(f"Missing: {meta}. Run data_loader_15dim.py first.")
    return json.loads(meta.read_text(encoding="utf-8")).get("features_15", [])


def _read_flat(feature_names: list[str]) -> list[dict]:
    flat = PKG_ROOT / "data" / "processed" / "flat_15dim.csv"
    if not flat.exists():
        raise SystemExit(f"Missing: {flat}. Run data_loader_15dim.py first.")
    rows: list[dict] = []
    with flat.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            if rr.get("split") != "train":
                continue
            y = rr.get("y_true", "")
            if y == "":
                continue
            try:
                rr["_y"] = int(float(y))
            except Exception:
                continue
            try:
                rr["_round"] = int(float(rr.get("round", "nan")))
            except Exception:
                continue
            rr["_name"] = (rr.get("player_name") or "").strip()
            if not rr["_name"]:
                continue
            feats = []
            for fn in feature_names:
                try:
                    v = float(rr.get(fn, "nan"))
                except Exception:
                    v = float("nan")
                feats.append(v)
            rr["_x"] = np.asarray(feats, dtype=np.float32)
            rows.append(rr)
    return rows


def _zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def _zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x2 = np.where(np.isfinite(x), x, mu)
    return ((x2 - mu) / sd).astype(np.float32)


def _build_sequences(rows: list[dict], L: int) -> tuple[np.ndarray, np.ndarray]:
    by_player: dict[str, list[dict]] = {}
    for r in rows:
        by_player.setdefault(r["_name"], []).append(r)
    Xs: list[np.ndarray] = []
    ys: list[int] = []
    for rr in by_player.values():
        rr = sorted(rr, key=lambda x: x["_round"])
        for i in range(len(rr)):
            if i + 1 < L:
                continue
            win = rr[i + 1 - L : i + 1]
            Xs.append(np.stack([w["_x"] for w in win], axis=0))
            ys.append(int(rr[i]["_y"]))
    if not Xs:
        return np.zeros((0, L, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(Xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _split(y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    # stratified via sorting by label then shuffling chunks
    pos = idx[y == 1]
    neg = idx[y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(a)
        n_tr = int(round(n * 0.8))
        n_va = int(round(n * 0.1))
        return a[:n_tr], a[n_tr : n_tr + n_va], a[n_tr + n_va :]

    trp, vap, tep = cut(pos)
    trn, van, ten = cut(neg)
    tr = np.concatenate([trp, trn])
    va = np.concatenate([vap, van])
    te = np.concatenate([tep, ten])
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


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

    feature_names = _load_feature_names()
    rows = _read_flat(feature_names)
    if not rows:
        raise SystemExit("No training rows found in flat_15dim.csv")

    out_csv = PKG_ROOT / "tables" / "seq_len_results_15dim.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    candidates = [6, 8, 12, 15, 20, 24, 30, 36]

    results: list[dict] = []
    for L in candidates:
        X, y = _build_sequences(rows, L)
        if len(X) < 1000:
            continue
        mu, sd = _zscore_fit(X.reshape(-1, X.shape[-1]))
        Xn = _zscore_apply(X, mu, sd)

        tr, va, te = _split(y, seed=42)
        Xtr, Xva, Xte = Xn[tr].reshape(len(tr), -1), Xn[va].reshape(len(va), -1), Xn[te].reshape(len(te), -1)
        ytr, yva, yte = y[tr], y[va], y[te]

        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 42,
        }
        booster = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
        pva = booster.predict(Xva)
        thr = _best_thr_f1(yva, pva)
        pte = booster.predict(Xte)
        results.append(
            {
                "seq_len": L,
                "val_auc": float(roc_auc_score(yva, pva)),
                "val_f1@thr": float(f1_score(yva, (pva >= thr).astype(int))),
                "test_auc": float(roc_auc_score(yte, pte)),
                "test_f1@thr": float(f1_score(yte, (pte >= thr).astype(int))),
                "thr": thr,
                "n_train_seq": int(len(tr)),
                "n_val_seq": int(len(va)),
                "n_test_seq": int(len(te)),
            }
        )
        print("L=", L, "val_f1=", results[-1]["val_f1@thr"], "test_f1=", results[-1]["test_f1@thr"])

    if not results:
        raise SystemExit("No results produced.")

    # sort by val_f1
    results.sort(key=lambda r: r["val_f1@thr"], reverse=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()

