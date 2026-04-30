from __future__ import annotations

"""
Paper-aligned 20-D per-timestep feature builder.

This implements Table 1 feature schema described in `data/article/ACM双栏.pdf`:
  - 15 core features + 5 position one-hot encodings = 20 dims
  - Label: Top-25% within each (season, round) by total_points
  - Season-based splits (no shuffle): Train 2016–22, Val 2022–23, Test 2023–25, Holdout 2025–26

Inputs (shipped in upload/data/raw):
  - cleaned_merged_seasons_team_aggregated.csv   (2016–25)
  - merged_gw_25-26.csv                         (holdout season)

Outputs:
  - data/processed/flat_paper20d.csv
  - data/processed/flat_paper20d_meta.json

Notes / assumptions (because raw sources differ from the paper’s upstream API tables):
  - `opponent_strength`, `team_rank_relative`, `opponent_defense_weakness` are derived from
    within-season rolling team statistics computed from match results.
  - `recent_form` is player rolling mean of total_points over last 5 appearances.
  - `starts` is approximated from minutes (>=60 => 1 else 0) when explicit starts is absent.
  - `pos_GKP` is set to 0 (reserved) unless a dedicated signal exists upstream.
"""

import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[1]


PAPER_FEATURES_15 = [
    # 1) Player Base Performance (7)
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "bps",
    "yellow_cards",
    "red_cards",
    # 2) Opponent & Schedule Context (5)
    "was_home",
    "opponent_strength",
    "is_season_opener",
    "days_since_last_game",
    "season_cumulative_minutes",
    # 3) Dynamic Ability Correction (3)
    "team_rank_relative",
    "opponent_defense_weakness",
    "recent_form",
]

POS_FEATURES_5 = ["pos_DEF", "pos_FWD", "pos_GK", "pos_GKP", "pos_MID"]
FEATURES_20 = PAPER_FEATURES_15 + POS_FEATURES_5


def _read_rows(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            out.append(rr)
    return out


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _to_int(x: str) -> int | None:
    try:
        return int(float(x))
    except Exception:
        return None


def _parse_time(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    # kickoff_time in Vaastav dumps is usually ISO with Z.
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _season_bucket(season: str) -> str:
    # season string like "2016-17"
    if season == "2025-26":
        return "holdout"
    try:
        y = int(season.split("-")[0])
    except Exception:
        return "train"
    if 2016 <= y <= 2021:
        return "train"
    if y == 2022:
        return "val"
    if y in (2023, 2024):
        return "test"
    # 2025 (if any pre 25-26 rows) keep in test by paper definition 2023–25 combined
    if y == 2025:
        return "test"
    return "train"


@dataclass
class TeamState:
    pts: int = 0
    gf: int = 0
    ga: int = 0
    matches: int = 0


def main() -> None:
    hist_path = PKG_ROOT / "data" / "raw" / "cleaned_merged_seasons_team_aggregated.csv"
    hold_path = PKG_ROOT / "data" / "raw" / "merged_gw_25-26.csv"
    if not hist_path.exists():
        raise SystemExit(f"Missing: {hist_path}")
    if not hold_path.exists():
        raise SystemExit(f"Missing: {hold_path}")

    hist = _read_rows(hist_path)
    hold = _read_rows(hold_path)

    # Normalize minimal schema
    def norm(rr: dict, *, season: str) -> dict:
        out = dict(rr)
        out["season_x"] = season
        # unify key names
        if "name" not in out and "player_name" in out:
            out["name"] = out["player_name"]
        if "team_x" not in out and "team" in out:
            out["team_x"] = out["team"]
        return out

    hist2 = [norm(r, season=(r.get("season_x") or "").strip()) for r in hist if (r.get("season_x") or "").strip()]
    hold2 = [norm(r, season="2025-26") for r in hold]

    # Build team rolling stats within each season+round based on match outcomes.
    # We approximate opponent strength as opponent points-per-match rank at that round.
    # We approximate defense weakness as opponent goals-against per match up to that round.
    by_season_round_team: dict[tuple[str, int], dict[str, TeamState]] = defaultdict(dict)
    seasons = sorted({r["season_x"] for r in hist2} | {"2025-26"})

    def iter_rows(season: str) -> list[dict]:
        if season == "2025-26":
            return hold2
        return [r for r in hist2 if r["season_x"] == season]

    # Precompute per (season, round) match results per team using available scores.
    for season in seasons:
        rows = iter_rows(season)
        # group by round
        by_round: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            rd = _to_int(r.get("round", "") or r.get("GW", "") or "")
            if rd is None:
                continue
            by_round[rd].append(r)

        team_state: dict[str, TeamState] = defaultdict(TeamState)
        for rd in sorted(by_round):
            # snapshot BEFORE applying this round results
            by_season_round_team[(season, rd)] = {t: TeamState(**vars(s)) for t, s in team_state.items()}

            # apply results using (team_x, opponent_team) and scores if present.
            # Each player-row repeats the same fixture; we update states per fixture only once.
            seen_fix: set[tuple[str, int, str]] = set()
            for r in by_round[rd]:
                team = (r.get("team_x") or "").strip()
                opp = (r.get("opponent_team") or "").strip()
                if not team or not opp:
                    continue
                # use fixture id when available to avoid double-counting
                fx = r.get("fixture", "")
                key = (fx, rd, team)
                if key in seen_fix:
                    continue
                seen_fix.add(key)

                th = _to_int(r.get("team_h_score", "") or "")
                ta = _to_int(r.get("team_a_score", "") or "")
                was_home = _to_int(r.get("was_home", "") or "")
                if th is None or ta is None or was_home is None:
                    continue

                gf = th if was_home == 1 else ta
                ga = ta if was_home == 1 else th

                s = team_state[team]
                s.matches += 1
                s.gf += int(gf)
                s.ga += int(ga)
                if gf > ga:
                    s.pts += 3
                elif gf == ga:
                    s.pts += 1

    # Player rolling states
    last_time_by_player: dict[tuple[str, str], datetime] = {}
    cum_minutes_by_player: dict[tuple[str, str], float] = defaultdict(float)
    recent_pts_by_player: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=5))

    # For label Top25% within each (season, round)
    pts_by_sr: dict[tuple[str, int], list[float]] = defaultdict(list)

    out_rows: list[dict] = []

    def emit(r: dict) -> None:
        name = (r.get("name") or "").strip()
        if not name:
            return
        season = (r.get("season_x") or "").strip()
        rd = _to_int(r.get("round", "") or r.get("GW", "") or "")
        if rd is None:
            return
        tp = _to_float(r.get("total_points", "nan"))
        if not math.isfinite(tp):
            return

        kickoff = (r.get("kickoff_time") or "").strip()
        dt = _parse_time(kickoff) or datetime(1970, 1, 1, tzinfo=timezone.utc)

        pos = (r.get("position") or "").strip()
        team = (r.get("team_x") or "").strip()
        opp_team = (r.get("opponent_team") or "").strip()
        was_home = 1.0 if str(r.get("was_home", "")).strip() in {"1", "True", "true"} else 0.0

        # starts: prefer explicit if present, else minutes proxy
        if "starts" in r and str(r.get("starts", "")).strip() != "":
            starts = 1.0 if _to_int(r.get("starts", "") or "0") == 1 else 0.0
        else:
            mins = _to_float(r.get("minutes", "0"))
            starts = 1.0 if math.isfinite(mins) and mins >= 60.0 else 0.0

        keyp = (season, name)
        # days since last game (per player)
        last_dt = last_time_by_player.get(keyp)
        if last_dt is None:
            dslg = 30.0
        else:
            dslg = max(0.0, (dt - last_dt).total_seconds() / 86400.0)
        last_time_by_player[keyp] = dt

        # season cumulative minutes
        mins = _to_float(r.get("minutes", "0"))
        if not math.isfinite(mins):
            mins = 0.0
        cum_minutes_by_player[keyp] += float(mins)
        scmin = float(cum_minutes_by_player[keyp])

        # recent form: rolling mean of last 5 total_points (excluding current)
        dq = recent_pts_by_player[keyp]
        # IMPORTANT: do NOT use current total_points as a feature (leakage).
        # If there is no history yet, fall back to 0.0.
        recent_form = float(sum(dq) / len(dq)) if len(dq) else 0.0
        dq.append(float(tp))

        # is season opener: first 3 rounds flag
        is_open = 1.0 if rd <= 3 else 0.0

        # opponent_strength and defense weakness from team snapshot BEFORE this round
        snap = by_season_round_team.get((season, rd), {})
        # rank teams by pts per match (avoid div0)
        def ppm(s: TeamState) -> float:
            return (s.pts / max(1, s.matches)) if s else 0.0

        teams_sorted = sorted(snap.items(), key=lambda kv: ppm(kv[1]), reverse=True)
        team_to_rank = {t: i + 1 for i, (t, _s) in enumerate(teams_sorted)}
        # relative rank vs league average rank
        if team and team in team_to_rank:
            avg_rank = (len(team_to_rank) + 1) / 2.0 if team_to_rank else 10.0
            team_rank_relative = float(team_to_rank[team] - avg_rank)
        else:
            team_rank_relative = 0.0

        if opp_team and opp_team in snap:
            opp_state = snap[opp_team]
            opponent_strength = float(ppm(opp_state))
            opponent_def_weak = float((opp_state.ga / max(1, opp_state.matches)))
        else:
            opponent_strength = 0.0
            opponent_def_weak = 0.0

        # position one-hot (5)
        pos_DEF = 1.0 if pos == "DEF" else 0.0
        pos_FWD = 1.0 if pos == "FWD" else 0.0
        pos_GK = 1.0 if pos == "GK" else 0.0
        pos_MID = 1.0 if pos == "MID" else 0.0
        pos_GKP = 0.0  # reserved

        row = {
            "player_name": name,
            "season": season,
            "round": int(rd),
            "kickoff_time": kickoff,
            "position": pos,
            "team": team,
            "total_points": float(tp),
            "split": _season_bucket(season),
            # 20-D features
            "minutes": float(mins),
            "goals_scored": _to_float(r.get("goals_scored", "0")),
            "assists": _to_float(r.get("assists", "0")),
            "clean_sheets": _to_float(r.get("clean_sheets", "0")),
            "bps": _to_float(r.get("bps", "0")),
            "yellow_cards": _to_float(r.get("yellow_cards", "0")),
            "red_cards": _to_float(r.get("red_cards", "0")),
            "was_home": float(was_home),
            "opponent_strength": float(opponent_strength),
            "is_season_opener": float(is_open),
            "days_since_last_game": float(dslg),
            "season_cumulative_minutes": float(scmin),
            "team_rank_relative": float(team_rank_relative),
            "opponent_defense_weakness": float(opponent_def_weak),
            "recent_form": float(recent_form),
            "pos_DEF": float(pos_DEF),
            "pos_FWD": float(pos_FWD),
            "pos_GK": float(pos_GK),
            "pos_GKP": float(pos_GKP),
            "pos_MID": float(pos_MID),
            # label to fill later
            "y_true": "",
        }
        out_rows.append(row)
        pts_by_sr[(season, int(rd))].append(float(tp))

    # Sort per season/player/time for stable rolling stats
    def sort_key(r: dict) -> tuple:
        season = (r.get("season_x") or "").strip()
        rd = _to_int(r.get("round", "") or r.get("GW", "") or "") or 0
        name = (r.get("name") or "").strip()
        t = _parse_time(r.get("kickoff_time") or "") or datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (season, name, rd, t)

    for r in sorted(hist2, key=sort_key):
        emit(r)
    for r in sorted(hold2, key=sort_key):
        emit(r)

    # Fill y_true: Top-25% within each (season, round)
    thr_by_sr: dict[tuple[str, int], float] = {}
    for key, lst in pts_by_sr.items():
        if not lst:
            continue
        srt = sorted(lst, reverse=True)
        k = max(1, int(math.ceil(0.25 * len(srt))))
        thr_by_sr[key] = float(srt[k - 1])

    for r in out_rows:
        thr = thr_by_sr.get((r["season"], int(r["round"])))
        if thr is None:
            r["y_true"] = ""
        else:
            r["y_true"] = 1 if float(r["total_points"]) >= thr else 0

    out_flat = PKG_ROOT / "data" / "processed" / "flat_paper20d.csv"
    out_meta = PKG_ROOT / "data" / "processed" / "flat_paper20d_meta.json"
    out_flat.parent.mkdir(parents=True, exist_ok=True)

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
    ] + FEATURES_20

    with out_flat.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in header})

    out_meta.write_text(
        json.dumps(
            {
                "features_20": FEATURES_20,
                "n_rows": len(out_rows),
                "splits": {"train": "2016-17..2021-22", "val": "2022-23", "test": "2023-24..2024-25", "holdout": "2025-26"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Wrote:", out_flat)
    print("Wrote:", out_meta)


if __name__ == "__main__":
    main()

