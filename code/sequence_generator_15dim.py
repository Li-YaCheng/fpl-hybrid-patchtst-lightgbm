from __future__ import annotations

"""
Sequence generator for the 15-dim feature set (reconstructed).

Outputs:
- artifacts/npz/processed_data_v2_L30_padpred.npz (default)
  contains:
    X_train, y_train, X_val, y_val, X_test, y_test, X_predict, y_predict (if available)
    feature_names, seq_len
- data/processed/predict_25_26_sequences_meta.csv
  row-aligned metadata for X_predict sequences (player_name, season, round, kickoff_time, total_points, y_true)

Assumptions:
- Input flat CSV is produced by code/data_loader_15dim.py
- Split is done by (train seasons) random stratified split into train/val/test.
  (This mirrors typical practice; if you need a specific split, replace split logic.)
"""

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FlatRow:
    player_name: str
    season: str
    round: int
    kickoff_time: str
    total_points: float
    y_true: int | None
    split: str
    feats: np.ndarray  # shape (d,)


def _read_flat(path: Path, feature_names: list[str]) -> list[FlatRow]:
    rows: list[FlatRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            name = (rr.get("player_name") or "").strip()
            if not name:
                continue
            season = (rr.get("season") or "").strip()
            try:
                rd = int(float(rr.get("round", "nan")))
            except Exception:
                continue
            kickoff = (rr.get("kickoff_time") or "").strip()
            try:
                pts = float(rr.get("total_points", "nan"))
            except Exception:
                continue
            if not math.isfinite(pts):
                continue
            y_raw = (rr.get("y_true") or "").strip()
            y: int | None
            if y_raw == "":
                y = None
            else:
                try:
                    y = int(float(y_raw))
                except Exception:
                    y = None
            split = (rr.get("split") or "").strip()
            feats = []
            for fn in feature_names:
                try:
                    v = float(rr.get(fn, "nan"))
                except Exception:
                    v = float("nan")
                feats.append(v)
            rows.append(
                FlatRow(
                    player_name=name,
                    season=season,
                    round=rd,
                    kickoff_time=kickoff,
                    total_points=float(pts),
                    y_true=y,
                    split=split,
                    feats=np.asarray(feats, dtype=np.float32),
                )
            )
    return rows


def _zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    # If a feature column is entirely NaN, both mu and sd can be NaN.
    # Force a safe normalization: mu=0, sd=1 for non-finite columns.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def _zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x2 = x.copy()
    x2 = np.where(np.isfinite(x2), x2, mu)
    return ((x2 - mu) / sd).astype(np.float32)


def _split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # stratified split: 80/10/10
    rng = random.Random(seed)
    idx_pos = [i for i, v in enumerate(y.tolist()) if v == 1]
    idx_neg = [i for i, v in enumerate(y.tolist()) if v == 0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def cut(lst: list[int]) -> tuple[list[int], list[int], list[int]]:
        n = len(lst)
        n_tr = int(round(n * 0.8))
        n_va = int(round(n * 0.1))
        tr = lst[:n_tr]
        va = lst[n_tr : n_tr + n_va]
        te = lst[n_tr + n_va :]
        return tr, va, te

    tr_p, va_p, te_p = cut(idx_pos)
    tr_n, va_n, te_n = cut(idx_neg)
    tr = np.asarray(tr_p + tr_n, dtype=np.int64)
    va = np.asarray(va_p + va_n, dtype=np.int64)
    te = np.asarray(te_p + te_n, dtype=np.int64)
    rng.shuffle(tr.tolist())
    rng.shuffle(va.tolist())
    rng.shuffle(te.tolist())
    return tr, va, te


def main() -> None:
    seq_len = 30
    seed = 42

    flat_path = PKG_ROOT / "data" / "processed" / "flat_15dim.csv"
    meta_path = PKG_ROOT / "data" / "processed" / "flat_15dim_meta.json"
    out_npz = PKG_ROOT / "artifacts" / "npz" / f"processed_data_v2_L{seq_len}_padpred.npz"
    out_meta = PKG_ROOT / "data" / "processed" / "predict_25_26_sequences_meta.csv"

    if not meta_path.exists():
        raise SystemExit(f"Missing meta: {meta_path}. Run data_loader_15dim.py first.")
    if not flat_path.exists():
        raise SystemExit(f"Missing flat: {flat_path}. Run data_loader_15dim.py first.")

    feature_names = json.loads(meta_path.read_text(encoding="utf-8")).get("features_15", [])
    if not feature_names:
        raise SystemExit("No feature_names in meta.")

    rows = _read_flat(flat_path, feature_names)

    # group by player within split
    by_player: dict[tuple[str, str], list[FlatRow]] = {}
    for r in rows:
        by_player.setdefault((r.player_name, r.split), []).append(r)

    def build_sequences(group_rows: list[FlatRow]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        rr = sorted(group_rows, key=lambda x: x.round)
        Xs: list[np.ndarray] = []
        ys: list[int] = []
        metas: list[dict] = []
        for i in range(len(rr)):
            if i + 1 < seq_len:
                continue
            window = rr[i + 1 - seq_len : i + 1]
            x = np.stack([w.feats for w in window], axis=0)  # (L,d)
            y = rr[i].y_true
            if y is None:
                yv = -1
            else:
                yv = int(y)
            Xs.append(x)
            ys.append(yv)
            metas.append(
                {
                    "player_name": rr[i].player_name,
                    "season": rr[i].season,
                    "round": rr[i].round,
                    "kickoff_time": rr[i].kickoff_time,
                    "total_points": rr[i].total_points,
                    "y_true": rr[i].y_true if rr[i].y_true is not None else "",
                }
            )
        if not Xs:
            return np.zeros((0, seq_len, len(feature_names)), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
        return np.stack(Xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64), metas

    X_train_list, y_train_list = [], []
    X_hold_list, y_hold_list, meta_hold_all = [], [], []

    for (_name, split), rr in by_player.items():
        X, y, metas = build_sequences(rr)
        if split == "train":
            if len(X):
                X_train_list.append(X)
                y_train_list.append(y)
        elif split == "holdout":
            if len(X):
                X_hold_list.append(X)
                y_hold_list.append(y)
                meta_hold_all.extend(metas)

    if not X_train_list:
        raise SystemExit("No training sequences built. Check flat_15dim.csv content.")

    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)

    # filter invalid labels (if any)
    m = y_train_all >= 0
    X_train_all = X_train_all[m]
    y_train_all = y_train_all[m]

    # normalize using training only
    mu, sd = _zscore_fit(X_train_all.reshape(-1, X_train_all.shape[-1]))
    X_train_all = _zscore_apply(X_train_all, mu, sd)

    tr_idx, va_idx, te_idx = _split_indices(y_train_all, seed=seed)
    X_tr, y_tr = X_train_all[tr_idx], y_train_all[tr_idx]
    X_va, y_va = X_train_all[va_idx], y_train_all[va_idx]
    X_te, y_te = X_train_all[te_idx], y_train_all[te_idx]

    # predict/holdout
    if X_hold_list:
        X_pred = np.concatenate(X_hold_list, axis=0)
        y_pred = np.concatenate(y_hold_list, axis=0)
        X_pred = _zscore_apply(X_pred, mu, sd)
    else:
        X_pred = np.zeros((0, seq_len, len(feature_names)), dtype=np.float32)
        y_pred = np.zeros((0,), dtype=np.int64)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        X_predict=X_pred,
        y_predict=y_pred,
        feature_names=np.asarray(feature_names, dtype=object),
        seq_len=int(seq_len),
        z_mu=mu,
        z_sd=sd,
    )

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["player_name", "season", "round", "kickoff_time", "total_points", "y_true"],
        )
        w.writeheader()
        for mrow in meta_hold_all:
            w.writerow(mrow)

    print("Wrote:", out_npz)
    print("Wrote:", out_meta)
    print("Train/Val/Test:", len(X_tr), len(X_va), len(X_te), "Predict:", len(X_pred))


if __name__ == "__main__":
    main()

