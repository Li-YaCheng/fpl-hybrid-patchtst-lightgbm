from __future__ import annotations

"""
15-dim data loader (reconstructed).

Goal:
- Read already-merged/cleaned gameweek-level CSVs
- Build a flat supervised dataset with a consistent schema
- Save to `upload/data/processed/flat_15dim.csv`

This is designed to be the first step of the end-to-end pipeline starting from merged CSVs
(as used in the original experiments).
"""

import csv
import json
import math
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[1]


FEATURES_15 = [
    # common FPL per-gameweek signals available in merged_gw_25-26.csv
    "xP",
    "minutes",
    "starts",
    "value",
    "selected",
    "transfers_balance",
    "creativity",
    "influence",
    "threat",
    "ict_index",
    "bps",
    "bonus",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
]


def _read_csv_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            rows.append(rr)
    return rows


def _safe_float(x: str) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    raw_holdout = PKG_ROOT / "data" / "raw" / "merged_gw_25-26.csv"
    raw_hist = PKG_ROOT / "data" / "raw" / "cleaned_merged_seasons.csv"
    out_flat = PKG_ROOT / "data" / "processed" / "flat_15dim.csv"
    out_meta = PKG_ROOT / "data" / "processed" / "flat_15dim_meta.json"

    if not raw_holdout.exists():
        raise SystemExit(f"Missing: {raw_holdout}")
    if not raw_hist.exists():
        raise SystemExit(f"Missing: {raw_hist}")

    # We build a single flat dataset with a split flag:
    # - split='train' for seasons before 2025-26 (from cleaned_merged_seasons.csv)
    # - split='holdout' for 2025-26 (from merged_gw_25-26.csv)
    #
    # Note: `cleaned_merged_seasons.csv` schema may differ across snapshots. We only rely on:
    # - player_name / name
    # - season / season_x
    # - round / GW
    # - total_points
    #
    # For missing features in the historical file, we write NaN; users may adapt mapping as needed.

    hist_rows = _read_csv_rows(raw_hist)
    hold_rows = _read_csv_rows(raw_holdout)

    header = [
        "player_name",
        "season",
        "round",
        "kickoff_time",
        "position",
        "team",
        "total_points",
        "y_true",
        "split",
    ] + FEATURES_15

    out: list[list] = []
    # collect points per (season, round) to compute Top25% label
    pts_by_sr: dict[tuple[str, int], list[float]] = {}

    def emit(rr: dict, split: str) -> None:
        name = (rr.get("player_name") or rr.get("name") or "").strip()
        if not name:
            return
        season = (rr.get("season_x") or rr.get("season") or ("2025-26" if split == "holdout" else "")).strip()
        rd = rr.get("round") or rr.get("GW") or rr.get("gw") or ""
        try:
            rd_i = int(float(rd))
        except Exception:
            return
        pts = _safe_float(rr.get("total_points", "nan"))
        if not math.isfinite(pts):
            return

        kickoff = (rr.get("kickoff_time") or "").strip()
        pos = (rr.get("position") or "").strip()
        team = (rr.get("team") or "").strip()

        # y_true will be computed after collecting all points within each (season, round)
        row = [name, season, rd_i, kickoff, pos, team, pts, "", split]
        for f in FEATURES_15:
            row.append(_safe_float(rr.get(f, "nan")))
        out.append(row)
        pts_by_sr.setdefault((season, rd_i), []).append(float(pts))

    for rr in hist_rows:
        # best-effort: exclude holdout season if it appears here
        season = (rr.get("season_x") or rr.get("season") or "").strip()
        if season == "2025-26":
            continue
        emit(rr, "train")

    for rr in hold_rows:
        emit(rr, "holdout")

    # compute Top25% threshold per (season, round) and fill y_true
    thr_by_sr: dict[tuple[str, int], float] = {}
    for key, pts_list in pts_by_sr.items():
        if not pts_list:
            continue
        pts_sorted = sorted(pts_list, reverse=True)
        k = max(1, int(math.ceil(0.25 * len(pts_sorted))))
        thr_by_sr[key] = float(pts_sorted[k - 1])

    # fill y_true column (index 7)
    for row in out:
        season = str(row[1])
        rd_i = int(row[2])
        pts = float(row[6])
        thr = thr_by_sr.get((season, rd_i))
        if thr is None:
            row[7] = ""
        else:
            row[7] = 1 if pts >= thr else 0

    _write_csv(out_flat, header, out)
    out_meta.write_text(json.dumps({"features_15": FEATURES_15, "n_rows": len(out)}, indent=2), encoding="utf-8")
    print("Wrote:", out_flat)
    print("Wrote:", out_meta)


if __name__ == "__main__":
    main()

