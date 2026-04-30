from __future__ import annotations

"""
Generate paper Table 6: Top-5 recommendation examples from 2025–26.

Paper (ACM双栏.pdf) shows example rows for GW28–29 with:
  Rank, Player, GW, p_hat, Pts, Hit

This script:
- reads the holdout prediction CSV (produced by `run_predict_holdout_paper.py`)
- selects top-K by predicted probability per gameweek
- writes:
  - tables/table6_holdout_top5_gw28_29.csv
  - tables/table6_holdout_top5_all_gameweeks.csv
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Row:
    player_name: str
    gw: int
    p_hat: float
    pts: float
    hit: int


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: str) -> int | None:
    try:
        return int(float(x))
    except Exception:
        return None


def _read_pred(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            name = (rr.get("player_name") or "").strip()
            if not name:
                continue
            gw = _safe_int(rr.get("round", "") or rr.get("GW", "") or "")
            if gw is None:
                continue
            p = _safe_float(rr.get("hybrid_proba", rr.get("proba", "")) or "")
            pts = _safe_float(rr.get("total_points", "") or "")
            y = rr.get("y_true", rr.get("is_high_score", "")).strip()
            hit = int(float(y)) if y != "" else 0
            if not math.isfinite(p):
                continue
            rows.append(Row(player_name=name, gw=int(gw), p_hat=float(p), pts=float(pts) if math.isfinite(pts) else float("nan"), hit=int(hit)))
    return rows


def _topk_by_gw(rows: list[Row], *, k: int) -> list[Row]:
    by_gw: dict[int, list[Row]] = {}
    for r in rows:
        by_gw.setdefault(r.gw, []).append(r)
    out: list[Row] = []
    for gw in sorted(by_gw):
        rr = sorted(by_gw[gw], key=lambda x: x.p_hat, reverse=True)[:k]
        out.extend(rr)
    return out


def _write_table6(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "Player", "GW", "p_hat", "Pts", "Hit"])
        # write grouped by GW with per-GW ranks
        by_gw: dict[int, list[Row]] = {}
        for r in rows:
            by_gw.setdefault(r.gw, []).append(r)
        for gw in sorted(by_gw):
            rr = sorted(by_gw[gw], key=lambda x: x.p_hat, reverse=True)
            for i, r0 in enumerate(rr, start=1):
                w.writerow([i, r0.player_name, gw, f"{r0.p_hat:.3f}", "" if not math.isfinite(r0.pts) else int(r0.pts) if abs(r0.pts - round(r0.pts)) < 1e-9 else f"{r0.pts:.1f}", int(r0.hit)])


def main() -> None:
    pred_csv = PKG_ROOT / "data" / "predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv"
    if not pred_csv.exists():
        raise SystemExit(f"Missing: {pred_csv}. Run run_predict_holdout_paper.py first.")

    rows = _read_pred(pred_csv)
    # All gameweeks top-5
    top5_all = _topk_by_gw(rows, k=5)
    _write_table6(PKG_ROOT / "tables" / "table6_holdout_top5_all_gameweeks.csv", top5_all)

    # Paper example: GW28–29 top-5
    top5_28_29 = [r for r in top5_all if r.gw in (28, 29)]
    _write_table6(PKG_ROOT / "tables" / "table6_holdout_top5_gw28_29.csv", top5_28_29)

    print("Wrote:", PKG_ROOT / "tables" / "table6_holdout_top5_all_gameweeks.csv")
    print("Wrote:", PKG_ROOT / "tables" / "table6_holdout_top5_gw28_29.csv")


if __name__ == "__main__":
    main()

