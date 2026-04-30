from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    player_name: str
    total_points: float
    y_true: int


def _svg_header(w: int, h: int) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>\n"


def _svg_footer() -> str:
    return "</svg>\n"


def _read_samples(pred_csv: Path) -> list[Sample]:
    out: list[Sample] = []
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        # Some CSVs are UTF-8 with BOM, yielding a header like "\ufeffplayer_name".
        name_key = "player_name"
        if r.fieldnames and name_key not in r.fieldnames:
            for k in r.fieldnames:
                if k and k.lstrip("\ufeff") == name_key:
                    name_key = k
                    break
        for row in r:
            name = (row.get(name_key) or "").strip()
            if not name:
                continue
            try:
                pts = float(row.get("total_points", "nan"))
            except Exception:
                pts = float("nan")
            try:
                y = int(float(row.get("y_true", "0")))
            except Exception:
                y = 0
            out.append(Sample(player_name=name, total_points=pts, y_true=y))
    return out


def _read_position_lookup(cleaned_players_csv: Path) -> dict[str, str]:
    """
    Build mapping: "First Second" -> element_type (GK/DEF/MID/FWD).
    Note: this is season-level roster metadata; we use it only for grouping.
    """
    m: dict[str, str] = {}
    with cleaned_players_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = (row.get("first_name") or "").strip()
            sn = (row.get("second_name") or "").strip()
            pos = (row.get("element_type") or "").strip()
            if not fn or not sn or not pos:
                continue
            key = f"{fn} {sn}"
            # Keep first occurrence; duplicates are rare and usually identical.
            if key not in m:
                m[key] = pos
    return m


def _nice_step(raw_step: float) -> float:
    if not math.isfinite(raw_step) or raw_step <= 0:
        return 1.0
    exp = math.floor(math.log10(raw_step))
    f = raw_step / (10**exp)
    if f > 5:
        nf = 10
    elif f > 2:
        nf = 5
    elif f > 1:
        nf = 2
    else:
        nf = 1
    return nf * (10**exp)


def _linear_ticks_zero_max(data_max: float, *, headroom: float = 1.10, target_ticks: int = 5) -> tuple[list[float], float]:
    lo = data_max * headroom if math.isfinite(data_max) and data_max > 0 else 1.0
    if not math.isfinite(lo) or lo <= 0:
        return [0.0, 1.0], 1.0
    rough = lo / max(target_ticks - 1, 1)
    step = _nice_step(rough)
    n = int(math.ceil(lo / step - 1e-12))
    axis_max = n * step
    ticks = [i * step for i in range(n + 1)]
    if len(ticks) < 2:
        ticks = [0.0, axis_max]
    return ticks, axis_max


def _fmt_rate(x: float) -> str:
    if not math.isfinite(x):
        return "NA"
    if abs(x) < 1e-12:
        return "0"
    if abs(x - 1) < 1e-12:
        return "1"
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _write_fig_top25_rate_by_position_svg(
    samples: list[Sample],
    pos_lookup: dict[str, str],
    out_path: Path,
) -> None:
    # Typography (user-tuned):
    # - y tick numerals: keep 24–26pt (do NOT inflate)
    # - axis title: 1.5× and leave more left margin
    # - bar value labels: 1.5×
    # - category labels: 1.5×
    # - "n=..." labels: ≥2× but not larger than category labels
    # - "overall": 2×
    fs_title = 32
    fs_tick = 25
    fs_axis = 36  # 1.5× (was 24)
    fs_bar_val = 36  # 1.5× (was 24-ish)
    fs_cat = 36  # 1.5× (was 24)
    fs_n = 36  # 2× (was 18) and <= fs_cat
    fs_overall = 36  # 2× (was 18)

    order = ["GK", "DEF", "MID", "FWD", "UNK"]
    stats: dict[str, list[int]] = {k: [0, 0] for k in order}  # [n, pos]
    for s in samples:
        pos = pos_lookup.get(s.player_name, "UNK")
        if pos not in stats:
            pos = "UNK"
        stats[pos][0] += 1
        stats[pos][1] += int(s.y_true == 1)

    rates: dict[str, float] = {}
    for pos in order:
        n, p = stats[pos]
        rates[pos] = (p / n) if n > 0 else float("nan")

    overall_rate = sum(int(s.y_true == 1) for s in samples) / len(samples) if samples else float("nan")
    data_max = max([r for r in rates.values() if math.isfinite(r)] + ([overall_rate] if math.isfinite(overall_rate) else [])) or 0.0
    ticks, axis_max = _linear_ticks_zero_max(data_max, headroom=1.18, target_ticks=5)
    tick_step = ticks[1] - ticks[0] if len(ticks) > 1 else axis_max

    w, h = 1200, 640
    pad_l, pad_r, pad_t, pad_b = 130, 40, 78, 110
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    x0 = pad_l
    y0 = pad_t + plot_h
    x1 = x0 + plot_w
    y1 = pad_t

    def sy(v: float) -> float:
        v = 0.0 if v < 0 else v
        v = axis_max if v > axis_max else v
        return y0 - (v / axis_max) * plot_h if axis_max > 0 else y0

    n_cat = len(order)
    gap = 30
    bw = (plot_w - gap * (n_cat + 1)) / n_cat

    lines: list[str] = []
    lines.append(_svg_header(w, h))
    lines.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='white'/>\n")
    lines.append(
        "<text x='{x}' y='46' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' fill='#111'>"
        "Exploratory analysis: Top25% rate by position</text>\n".format(x=w / 2, fs=fs_title)
    )
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#222' stroke-width='1.6'/>\n")
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#222' stroke-width='1.6'/>\n")

    # y grid + ticks
    for tv in ticks:
        yy = sy(tv)
        lines.append(f"<line x1='{x0}' y1='{yy:.2f}' x2='{x1}' y2='{yy:.2f}' stroke='#e8e8e8' stroke-width='1'/>\n")
        lines.append(f"<line x1='{x0-8}' y1='{yy:.2f}' x2='{x0}' y2='{yy:.2f}' stroke='#222' stroke-width='1.35'/>\n")
        lab = _fmt_rate(tv)
        lines.append(
            "<text x='{x}' y='{y:.2f}' text-anchor='end' font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                x=x0 - 14, y=yy + 8, fs=fs_tick, lab=lab
            )
        )

    # overall mean line
    if math.isfinite(overall_rate):
        yy = sy(overall_rate)
        lines.append(
            f"<line x1='{x0}' y1='{yy:.2f}' x2='{x1}' y2='{yy:.2f}' stroke='#999' stroke-width='2.25' stroke-dasharray='8,6'/>\n"
        )
        lines.append(
            "<text x='{x}' y='{y:.2f}' text-anchor='start' font-family='Arial' font-size='{fs}' fill='#666'>overall</text>\n".format(
                x=x1 - 220, y=yy - 14, fs=fs_overall
            )
        )

    # bars
    for i, pos in enumerate(order):
        v = rates[pos]
        xv = x0 + gap + i * (bw + gap)
        hh = 0.0 if not math.isfinite(v) else (max(0.0, v) / axis_max) * plot_h if axis_max > 0 else 0.0
        yv = y0 - hh
        fill = "#2F4B7C" if pos != "UNK" else "#999999"
        stroke = "#1f355f" if pos != "UNK" else "#777777"
        lines.append(
            "<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{h:.2f}' fill='{fill}' opacity='0.92' stroke='{stroke}' stroke-width='1'/>\n".format(
                x=xv, y=yv, w=bw, h=hh, fill=fill, stroke=stroke
            )
        )
        # value label
        if math.isfinite(v):
            lines.append(
                "<text x='{x:.2f}' y='{y:.2f}' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' fill='#111'>{t}</text>\n".format(
                    x=xv + bw / 2, y=yv - 12, fs=fs_bar_val, t=_fmt_rate(v)
                )
            )
        # category + N
        n, _p = stats[pos]
        lines.append(
            "<text x='{x:.2f}' y='{y}' text-anchor='middle' font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                x=xv + bw / 2, y=y0 + 42, fs=fs_cat, lab=pos
            )
        )
        lines.append(
            "<text x='{x:.2f}' y='{y}' text-anchor='middle' font-family='Arial' font-size='{fs}' fill='#555'>n={n}</text>\n".format(
                x=xv + bw / 2, y=y0 + 82, fs=fs_n, n=n
            )
        )

    # axis label (only y; avoid bottom caption per your preference)
    lines.append(
        "<text x='{x}' y='{y:.2f}' transform='rotate(-90 {x} {y:.2f})' text-anchor='middle' font-family='Arial' "
        "font-size='{fs}' fill='#222'>Top25% rate</text>\n".format(x=54, y=pad_t + plot_h / 2, fs=fs_axis)
    )
    lines.append(_svg_footer())
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_table_position_summary(
    samples: list[Sample],
    pos_lookup: dict[str, str],
    out_csv: Path,
    out_tex: Path,
) -> None:
    order = ["GK", "DEF", "MID", "FWD", "UNK"]
    rows: list[tuple[str, int, float, float]] = []
    for pos in order:
        pts: list[float] = []
        ys: list[int] = []
        for s in samples:
            p = pos_lookup.get(s.player_name, "UNK")
            if p != pos:
                continue
            if math.isfinite(s.total_points):
                pts.append(s.total_points)
            ys.append(int(s.y_true == 1))
        n = len(ys)
        mean_pts = (sum(pts) / len(pts)) if pts else float("nan")
        rate = (sum(ys) / n) if n > 0 else float("nan")
        rows.append((pos, n, mean_pts, rate))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "n_samples", "mean_points", "top25_rate"])
        for pos, n, mp, rr in rows:
            w.writerow([pos, n, f"{mp:.3f}" if math.isfinite(mp) else "", _fmt_rate(rr) if math.isfinite(rr) else ""])

    # Simple LaTeX table snippet (two-column friendly; no extra commentary)
    lines = []
    lines.append("\\begin{table}[t]\n")
    lines.append("\\centering\n")
    lines.append("\\small\n")
    lines.append("\\setlength{\\tabcolsep}{6pt}\n")
    lines.append("\\begin{tabular}{lrrr}\n")
    lines.append("\\toprule\n")
    lines.append("Position & $N$ & Mean points & Top25\\% rate \\\\\n")
    lines.append("\\midrule\n")
    for pos, n, mp, rr in rows:
        mp_s = f"{mp:.2f}" if math.isfinite(mp) else "--"
        rr_s = _fmt_rate(rr) if math.isfinite(rr) else "--"
        lines.append(f"{pos} & {n} & {mp_s} & {rr_s} \\\\\n")
    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
    lines.append("\\caption{Dataset summary by position after preprocessing.}\n")
    lines.append("\\label{tab:data_by_position}\n")
    lines.append("\\end{table}\n")
    out_tex.write_text("".join(lines), encoding="utf-8")


def _write_fig_points_hist_svg(samples: list[Sample], out_path: Path) -> None:
    fs_title = 32
    fs_tick = 25
    fs_axis = 24

    # Bins: 0..15 and a final overflow bin "15+"
    max_display = 15
    bins = list(range(0, max_display + 1))  # integer points
    counts = [0 for _ in range(max_display + 2)]  # 0..15 plus overflow
    for s in samples:
        if not math.isfinite(s.total_points):
            continue
        v = int(round(s.total_points))
        if v < 0:
            v = 0
        if v > max_display:
            counts[-1] += 1
        else:
            counts[v] += 1

    data_max = max(counts) if counts else 1
    ticks, axis_max = _linear_ticks_zero_max(float(data_max), headroom=1.12, target_ticks=5)

    w, h = 1200, 640
    pad_l, pad_r, pad_t, pad_b = 130, 40, 78, 110
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    x0 = pad_l
    y0 = pad_t + plot_h
    x1 = x0 + plot_w
    y1 = pad_t

    def sy(v: float) -> float:
        v = 0.0 if v < 0 else v
        v = axis_max if v > axis_max else v
        return y0 - (v / axis_max) * plot_h if axis_max > 0 else y0

    n_cat = len(counts)
    gap = 8
    bw = (plot_w - gap * (n_cat + 1)) / n_cat

    lines: list[str] = []
    lines.append(_svg_header(w, h))
    lines.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='white'/>\n")
    lines.append(
        "<text x='{x}' y='46' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' fill='#111'>"
        "Exploratory analysis: distribution of weekly points</text>\n".format(x=w / 2, fs=fs_title)
    )
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#222' stroke-width='1.6'/>\n")
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#222' stroke-width='1.6'/>\n")

    for tv in ticks:
        yy = sy(tv)
        lines.append(f"<line x1='{x0}' y1='{yy:.2f}' x2='{x1}' y2='{yy:.2f}' stroke='#e8e8e8' stroke-width='1'/>\n")
        lines.append(f"<line x1='{x0-8}' y1='{yy:.2f}' x2='{x0}' y2='{yy:.2f}' stroke='#222' stroke-width='1.35'/>\n")
        lab = str(int(tv)) if abs(tv - round(tv)) < 1e-9 else f"{tv:.0f}"
        lines.append(
            "<text x='{x}' y='{y:.2f}' text-anchor='end' font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                x=x0 - 14, y=yy + 8, fs=fs_tick, lab=lab
            )
        )

    for i, c in enumerate(counts):
        xv = x0 + gap + i * (bw + gap)
        hh = (c / axis_max) * plot_h if axis_max > 0 else 0.0
        yv = y0 - hh
        lines.append(
            "<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{h:.2f}' fill='#9ecae1' stroke='#6baed6' stroke-width='1'/>"
            "<title>count={c}</title>\n".format(x=xv, y=yv, w=bw, h=hh, c=c)
        )
        # x tick label (sparse): show 0,5,10,15,15+
        if i in (0, 5, 10, 15, 16):
            lab = "15+" if i == 16 else str(i)
            lines.append(
                "<text x='{x:.2f}' y='{y}' text-anchor='middle' font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                    x=xv + bw / 2, y=y0 + 48, fs=fs_tick, lab=lab
                )
            )

    lines.append(
        "<text x='{x}' y='{y:.2f}' transform='rotate(-90 {x} {y:.2f})' text-anchor='middle' font-family='Arial' "
        "font-size='{fs}' fill='#222'>Count</text>\n".format(x=32, y=pad_t + plot_h / 2, fs=fs_axis)
    )
    lines.append(_svg_footer())
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parent
    pred_csv = base / "predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv"
    cleaned_players = base.parent / "2025-26" / "cleaned_players.csv"
    out_dir = base / "pred_outputs_25_26_patchtst_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_samples(pred_csv)
    pos_lookup = _read_position_lookup(cleaned_players)

    fig1 = out_dir / "figEDA1_top25_rate_by_position.svg"
    fig2 = out_dir / "figEDA2_points_distribution.svg"
    tab_csv = out_dir / "tableEDA1_position_summary.csv"
    tab_tex = out_dir / "tableEDA1_position_summary.tex"

    _write_fig_top25_rate_by_position_svg(samples, pos_lookup, fig1)
    _write_table_position_summary(samples, pos_lookup, tab_csv, tab_tex)
    _write_fig_points_hist_svg(samples, fig2)

    print("Wrote:", fig1)
    print("Wrote:", tab_csv)
    print("Wrote:", tab_tex)
    print("Wrote:", fig2)
    print("Samples:", len(samples), "Position map:", len(pos_lookup))


if __name__ == "__main__":
    main()

