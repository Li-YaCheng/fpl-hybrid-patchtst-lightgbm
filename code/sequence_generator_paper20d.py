from __future__ import annotations

"""
Paper-aligned sequence generator for the 20-D feature set.

Implements:
- Sequence length L=30 by default (and supports L=15 for ablation)
- Season-based splits are already present in flat_paper20d.csv (no random shuffling)
- Z-score normalization fit on Train only; applied to Val/Test/Holdout

Outputs:
  - artifacts/npz/paper20d_L{L}.npz
    keys: X_train,y_train,X_val,y_val,X_test,y_test,X_predict,y_predict,feature_names,seq_len,z_mu,z_sd
"""

import csv
import json
import math
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
    feats: np.ndarray  # (d,)


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
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def _zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x2 = np.where(np.isfinite(x), x, mu)
    return ((x2 - mu) / sd).astype(np.float32)


def main() -> None:
    seq_len = int((PKG_ROOT / "SEQ_LEN.txt").read_text(encoding="utf-8").strip()) if (PKG_ROOT / "SEQ_LEN.txt").exists() else 30

    flat_path = PKG_ROOT / "data" / "processed" / "flat_paper20d.csv"
    meta_path = PKG_ROOT / "data" / "processed" / "flat_paper20d_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing: {meta_path}. Run data_loader_paper20d.py first.")
    if not flat_path.exists():
        raise SystemExit(f"Missing: {flat_path}. Run data_loader_paper20d.py first.")

    feature_names = json.loads(meta_path.read_text(encoding="utf-8")).get("features_20", [])
    if not feature_names:
        raise SystemExit("No feature_names in meta.")

    rows = _read_flat(flat_path, feature_names)
    by_player_split: dict[tuple[str, str], list[FlatRow]] = {}
    for r in rows:
        by_player_split.setdefault((r.player_name, r.split), []).append(r)

    def build_sequences(rr: list[FlatRow]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        rr2 = sorted(rr, key=lambda x: (x.season, x.round))
        Xs: list[np.ndarray] = []
        ys: list[int] = []
        metas: list[dict] = []
        for i in range(len(rr2)):
            # Predict label at time i using the PREVIOUS seq_len timesteps (i-seq_len .. i-1).
            # This prevents leakage of same-gameweek features into the label.
            if i < seq_len:
                continue
            window = rr2[i - seq_len : i]
            x = np.stack([w.feats for w in window], axis=0)
            y = rr2[i].y_true
            if y is None:
                continue
            Xs.append(x)
            ys.append(int(y))
            metas.append(
                {
                    "player_name": rr2[i].player_name,
                    "season": rr2[i].season,
                    "round": rr2[i].round,
                    "kickoff_time": rr2[i].kickoff_time,
                    "total_points": rr2[i].total_points,
                    "y_true": rr2[i].y_true,
                }
            )
        if not Xs:
            return (
                np.zeros((0, seq_len, len(feature_names)), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                [],
            )
        return np.stack(Xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64), metas

    buckets = {"train": [], "val": [], "test": [], "holdout": []}
    metas_holdout: list[dict] = []
    for (_p, split), rr in by_player_split.items():
        X, y, metas = build_sequences(rr)
        if split == "holdout":
            metas_holdout.extend(metas)
        if len(X):
            buckets[split].append((X, y))

    if not buckets["train"]:
        raise SystemExit("No train sequences built.")

    X_tr = np.concatenate([x for x, _y in buckets["train"]], axis=0)
    y_tr = np.concatenate([y for _x, y in buckets["train"]], axis=0)
    X_va = np.concatenate([x for x, _y in buckets["val"]], axis=0) if buckets["val"] else np.zeros((0, seq_len, len(feature_names)), dtype=np.float32)
    y_va = np.concatenate([y for _x, y in buckets["val"]], axis=0) if buckets["val"] else np.zeros((0,), dtype=np.int64)
    X_te = np.concatenate([x for x, _y in buckets["test"]], axis=0) if buckets["test"] else np.zeros((0, seq_len, len(feature_names)), dtype=np.float32)
    y_te = np.concatenate([y for _x, y in buckets["test"]], axis=0) if buckets["test"] else np.zeros((0,), dtype=np.int64)
    X_pr = np.concatenate([x for x, _y in buckets["holdout"]], axis=0) if buckets["holdout"] else np.zeros((0, seq_len, len(feature_names)), dtype=np.float32)
    y_pr = np.concatenate([y for _x, y in buckets["holdout"]], axis=0) if buckets["holdout"] else np.zeros((0,), dtype=np.int64)

    mu, sd = _zscore_fit(X_tr.reshape(-1, X_tr.shape[-1]))
    X_tr = _zscore_apply(X_tr, mu, sd)
    if len(X_va):
        X_va = _zscore_apply(X_va, mu, sd)
    if len(X_te):
        X_te = _zscore_apply(X_te, mu, sd)
    if len(X_pr):
        X_pr = _zscore_apply(X_pr, mu, sd)

    out_npz = PKG_ROOT / "artifacts" / "npz" / f"paper20d_L{seq_len}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        X_predict=X_pr,
        y_predict=y_pr,
        feature_names=np.asarray(feature_names, dtype=object),
        seq_len=int(seq_len),
        z_mu=mu,
        z_sd=sd,
    )

    meta_out = PKG_ROOT / "data" / "processed" / f"paper20d_sequences_holdout_meta_L{seq_len}.csv"
    with meta_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["player_name", "season", "round", "kickoff_time", "total_points", "y_true"])
        w.writeheader()
        for m in metas_holdout:
            w.writerow(m)

    print("Wrote:", out_npz)
    print("Wrote:", meta_out)
    print("Train/Val/Test:", len(X_tr), len(X_va), len(X_te), "Predict:", len(X_pr))


if __name__ == "__main__":
    main()

