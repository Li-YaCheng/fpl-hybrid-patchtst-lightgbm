from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    player_name: str
    round: int
    total_points: float
    y_true: int  # Top25% label
    proba: float


def _svg_header(w: int, h: int) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>\n"


def _svg_footer() -> str:
    return "</svg>\n"


def _quantile(sorted_x: list[float], q: float) -> float:
    if not sorted_x:
        return float("nan")
    q = float(q)
    q = min(1.0, max(0.0, q))
    k = (len(sorted_x) - 1) * q
    i = int(math.floor(k))
    j = int(math.ceil(k))
    if i == j:
        return float(sorted_x[i])
    return float(sorted_x[i] * (j - k) + sorted_x[j] * (k - i))


def _nice_step(raw_step: float) -> float:
    """1/2/5 * 10^n style step, similar to matplotlib 'nice' ticks."""
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


def _linear_ticks_zero_max(
    data_max: float, *, headroom: float = 1.12, target_ticks: int = 5
) -> tuple[list[float], float]:
    """
    Ticks from 0 to a readable 'nice' axis maximum >= data_max * headroom.
    Used for conference-style y-axes instead of only {0, vmax/2, vmax}.
    """
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


def _format_y_tick(tv: float, *, is_rate: bool, step: float) -> str:
    if is_rate:
        if abs(tv - 1.0) < 1e-9:
            return "1"
        if abs(tv) < 1e-9:
            return "0"
        dec = 3 if step < 0.05 else 2
        s = f"{tv:.{dec}f}".rstrip("0").rstrip(".")
        return s if s else "0"
    if abs(tv - round(tv)) < 1e-6 * max(1.0, abs(tv)):
        return str(int(round(tv)))
    dec = 2 if step < 0.2 else 1
    s = f"{tv:.{dec}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _read_pred_rows(pred_csv: Path) -> list[Row]:
    rows: list[Row] = []
    with pred_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # 兼容 UTF-8 BOM 等情况：统一 key
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            rows.append(
                Row(
                    player_name=str(rr["player_name"]),
                    round=int(rr["round"]),
                    total_points=float(rr["total_points"]),
                    y_true=int(rr.get("y_true", rr.get("is_high_score", 0))),
                    proba=float(rr["hybrid_proba"]),
                )
            )
    return rows


def _make_topk_curve(rows: list[Row], k: int = 10) -> list[tuple[int, int, float, float]]:
    """
    per round: (round, n_round, hit_rate@k, mean_points@k)
    """
    by_r: dict[int, list[Row]] = {}
    for r in rows:
        by_r.setdefault(r.round, []).append(r)
    out: list[tuple[int, int, float, float]] = []
    for rd in sorted(by_r):
        rr = by_r[rd]
        rr_sorted = sorted(rr, key=lambda x: x.proba, reverse=True)
        top = rr_sorted[: min(k, len(rr_sorted))]
        hit = sum(int(x.y_true == 1) for x in top)
        hit_rate = hit / len(top) if top else float("nan")
        mean_pts = sum(x.total_points for x in top) / len(top) if top else float("nan")
        out.append((rd, len(rr), float(hit_rate), float(mean_pts)))
    return out


def _make_calibration(rows: list[Row], bins: int = 10) -> list[tuple[float, float, int]]:
    """
    returns list of (bin_center_p, empirical_pos_rate, count)
    """
    bins = max(5, int(bins))
    counts = [0] * bins
    sum_p = [0.0] * bins
    sum_y = [0] * bins
    for r in rows:
        p = r.proba
        b = int(p * bins)
        if b >= bins:
            b = bins - 1
        if b < 0:
            b = 0
        counts[b] += 1
        sum_p[b] += p
        sum_y[b] += int(r.y_true == 1)
    out: list[tuple[float, float, int]] = []
    for b in range(bins):
        c = counts[b]
        if c <= 0:
            continue
        mp = sum_p[b] / c
        py = sum_y[b] / c
        out.append((float(mp), float(py), int(c)))
    return out


def _write_fig_topk_hit_svg(per_round: list[tuple[int, int, float, float]], out_path: Path, k: int) -> None:
    w, h = 900, 520
    pad_l, pad_r, pad_t, pad_b = 70, 30, 50, 60
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    rds = [x[0] for x in per_round]
    hits = [x[2] for x in per_round]  # 0..1
    pts = [x[3] for x in per_round]
    if not rds:
        return
    min_rd, max_rd = min(rds), max(rds)

    def sx(rd: int) -> float:
        if max_rd == min_rd:
            return pad_l + plot_w / 2
        return pad_l + (rd - min_rd) / (max_rd - min_rd) * plot_w

    def sy_hit(v: float) -> float:
        # hit rate always in [0,1]
        v = 0.0 if v < 0 else (1.0 if v > 1 else v)
        return pad_t + (1.0 - v) * plot_h

    # points for secondary axis scaling
    pts_ok = [p for p in pts if math.isfinite(p)]
    min_pts = min(pts_ok) if pts_ok else 0.0
    max_pts = max(pts_ok) if pts_ok else 1.0
    if max_pts <= min_pts:
        max_pts = min_pts + 1.0
    max_pts *= 1.05

    def sy_pts(v: float) -> float:
        return pad_t + (max_pts - v) / (max_pts - min_pts) * plot_h

    x0 = pad_l
    y0 = pad_t + plot_h
    x1 = pad_l + plot_w
    y1 = pad_t

    lines: list[str] = []
    lines.append(_svg_header(w, h))
    lines.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='white'/>\n")
    lines.append(
        "<text x='{x}' y='28' text-anchor='middle' font-family='Arial' font-size='18'>"
        "Hold-out 2025–26: Top-{k} recommendation quality by gameweek</text>\n".format(x=w / 2, k=k)
    )
    # axes (left for hit, right for points)
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#333' stroke-width='1.2'/>\n")
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333' stroke-width='1.2'/>\n")
    lines.append(f"<line x1='{x1}' y1='{y0}' x2='{x1}' y2='{y1}' stroke='#333' stroke-width='1.2'/>\n")

    # left y grid for hit rate
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = sy_hit(t)
        lines.append(f"<line x1='{x0}' y1='{yy:.2f}' x2='{x1}' y2='{yy:.2f}' stroke='#eee' stroke-width='1'/>\n")
        lines.append(
            "<text x='{x}' y='{y:.2f}' text-anchor='end' font-family='Arial' font-size='12'>{lab}</text>\n".format(
                x=x0 - 8, y=yy + 4, lab="{:.2f}".format(t)
            )
        )

    # right y ticks for points (3 ticks)
    for t in (min_pts, (min_pts + max_pts) / 2, max_pts):
        yy = sy_pts(t)
        lines.append(
            "<text x='{x}' y='{y:.2f}' text-anchor='start' font-family='Arial' font-size='12' fill='#2F4B7C'>{lab}</text>\n".format(
                x=x1 + 8, y=yy + 4, lab="{:.1f}".format(t)
            )
        )

    # x ticks every 4 gameweeks
    step = 4
    for rd in range(min_rd, max_rd + 1, step):
        xx = sx(rd)
        lines.append(f"<line x1='{xx:.2f}' y1='{y0}' x2='{xx:.2f}' y2='{y0 + 5}' stroke='#333'/>\n")
        lines.append(
            "<text x='{x:.2f}' y='{y}' text-anchor='middle' font-family='Arial' font-size='12'>{rd}</text>\n".format(
                x=xx, y=y0 + 22, rd=rd
            )
        )

    # hit-rate polyline (purple)
    pts_hit = " ".join("{:.2f},{:.2f}".format(sx(rd), sy_hit(v)) for rd, v in zip(rds, hits))
    lines.append(f"<polyline points='{pts_hit}' fill='none' stroke='#7A5195' stroke-width='2.5'/>\n")
    for rd, v in zip(rds, hits):
        lines.append(f"<circle cx='{sx(rd):.2f}' cy='{sy_hit(v):.2f}' r='3.5' fill='#7A5195'/>\n")

    # mean points polyline (blue)
    pts_pts = " ".join("{:.2f},{:.2f}".format(sx(rd), sy_pts(v)) for rd, v in zip(rds, pts))
    lines.append(f"<polyline points='{pts_pts}' fill='none' stroke='#2F4B7C' stroke-width='2.2' opacity='0.9'/>\n")
    for rd, v in zip(rds, pts):
        lines.append(f"<circle cx='{sx(rd):.2f}' cy='{sy_pts(v):.2f}' r='3.2' fill='#2F4B7C' opacity='0.9'/>\n")

    # labels
    lines.append(
        "<text x='{x:.2f}' y='{y}' text-anchor='middle' font-family='Arial' font-size='14'>Gameweek (round)</text>\n".format(
            x=pad_l + plot_w / 2, y=h - 18
        )
    )
    lines.append(
        "<text x='18' y='{y:.2f}' transform='rotate(-90 18 {y:.2f})' text-anchor='middle' font-family='Arial' font-size='14'>Hit rate @ Top-{k}</text>\n".format(
            y=pad_t + plot_h / 2, k=k
        )
    )
    lines.append(
        "<text x='{x}' y='{y:.2f}' transform='rotate(-90 {x} {y:.2f})' text-anchor='middle' font-family='Arial' font-size='14' fill='#2F4B7C'>Mean points @ Top-{k}</text>\n".format(
            x=w - 14, y=pad_t + plot_h / 2, k=k
        )
    )
    # legend
    lines.append(
        "<text x='{x}' y='{y}' font-family='Arial' font-size='12' fill='#7A5195'>● Hit rate</text>\n".format(
            x=pad_l + 10, y=pad_t + 20
        )
    )
    lines.append(
        "<text x='{x}' y='{y}' font-family='Arial' font-size='12' fill='#2F4B7C'>● Mean points</text>\n".format(
            x=pad_l + 120, y=pad_t + 20
        )
    )
    lines.append(_svg_footer())
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_fig_calibration_svg(cal: list[tuple[float, float, int]], out_path: Path) -> None:
    # Print/slide friendly; tick numerals emphasized (24–26 pt class → 25)
    fs_title = 32
    fs_tick = 25
    fs_y_axis = 24
    tick_dy = 8  # vertical nudge for tick label baselines

    w, h = 1200, 640
    pad_l, pad_r, pad_t, pad_b = 100, 40, 78, 72
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    x0 = pad_l
    y0 = pad_t + plot_h
    x1 = pad_l + plot_w
    y1 = pad_t

    def sx(v: float) -> float:
        v = 0.0 if v < 0 else (1.0 if v > 1 else v)
        return pad_l + v * plot_w

    def sy(v: float) -> float:
        v = 0.0 if v < 0 else (1.0 if v > 1 else v)
        return pad_t + (1.0 - v) * plot_h

    def _tick_lab(t: float) -> str:
        if abs(t - round(t)) < 1e-9:
            return str(int(round(t)))
        s = f"{t:.2f}".rstrip("0").rstrip(".")
        return s if s else "0"

    lines: list[str] = []
    lines.append(_svg_header(w, h))
    lines.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='white'/>\n")
    lines.append(
        "<text x='{x}' y='46' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' fill='#111'>"
        "Hold-out 2025–26: calibration</text>\n".format(x=w / 2, fs=fs_title)
    )
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#222' stroke-width='1.6'/>\n")
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#222' stroke-width='1.6'/>\n")

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx = sx(t)
        yy = sy(t)
        lines.append(f"<line x1='{xx:.2f}' y1='{y0}' x2='{xx:.2f}' y2='{y1}' stroke='#e8e8e8' stroke-width='1'/>\n")
        lines.append(f"<line x1='{x0}' y1='{yy:.2f}' x2='{x1}' y2='{yy:.2f}' stroke='#e8e8e8' stroke-width='1'/>\n")
        lines.append(
            "<text x='{x:.2f}' y='{y}' text-anchor='middle' font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                x=xx, y=y0 + 32 + tick_dy, fs=fs_tick, lab=_tick_lab(t)
            )
        )
        lines.append(
            "<text x='{x}' y='{y:.2f}' text-anchor='end' font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                x=x0 - 14, y=yy + tick_dy, fs=fs_tick, lab=_tick_lab(t)
            )
        )

    lines.append(
        "<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y1}' stroke='#999' stroke-width='2.25' stroke-dasharray='8,6'/>\n".format(
            x0=sx(0.0), y0=sy(0.0), x1=sx(1.0), y1=sy(1.0)
        )
    )

    cal_sorted = sorted(cal, key=lambda x: x[0])
    pts = " ".join("{:.2f},{:.2f}".format(sx(mp), sy(obs)) for mp, obs, _c in cal_sorted)
    lines.append(f"<polyline points='{pts}' fill='none' stroke='#4C78A8' stroke-width='3'/>\n")
    for mp, obs, c in cal_sorted:
        r = 5.5 + 14.0 * (min(1.0, math.sqrt(c) / 80.0))
        lines.append(
            "<circle cx='{x:.2f}' cy='{y:.2f}' r='{r:.2f}' fill='#4C78A8' opacity='0.88' stroke='#2f5f8a' stroke-width='1'/>"
            "<title>n={n}</title>\n".format(x=sx(mp), y=sy(obs), r=r, n=c)
        )

    lines.append(
        "<text x='{x}' y='{y:.2f}' transform='rotate(-90 {x} {y:.2f})' text-anchor='middle' font-family='Arial' font-size='{fs}' fill='#222'>"
        "Empirical Top25% rate</text>\n".format(x=32, y=pad_t + plot_h / 2, fs=fs_y_axis)
    )
    lines.append(_svg_footer())
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_fig_overall_uplift_svg(rows: list[Row], out_path: Path, k: int = 10) -> None:
    """
    Overall comparison: Top-K recommendations vs all samples.
    Plots:
      - mean points
      - hit rate (Top25% rate)
    """
    # aggregate
    all_mean_pts = sum(r.total_points for r in rows) / len(rows) if rows else float("nan")
    all_hit = sum(int(r.y_true == 1) for r in rows) / len(rows) if rows else float("nan")

    # Top-K per round (deployment-style)
    by_r: dict[int, list[Row]] = {}
    for r in rows:
        by_r.setdefault(r.round, []).append(r)
    top: list[Row] = []
    for rd in sorted(by_r):
        rr = sorted(by_r[rd], key=lambda x: x.proba, reverse=True)
        top.extend(rr[: min(k, len(rr))])
    top_mean_pts = sum(r.total_points for r in top) / len(top) if top else float("nan")
    top_hit = sum(int(r.y_true == 1) for r in top) / len(top) if top else float("nan")

    # Figure layout (print-friendly; fonts tuned per paper)
    fs_title = 32
    fs_panel = 24
    # Per user request:
    # - axis-related text (x/y ticks, axis titles, category labels): 1.5×
    # - bar value labels: +2 pt
    fs_tick = int(round(18 * 1.5))  # 27
    fs_axis = int(round(18 * 1.5))  # 27
    fs_cat = int(round(18 * 1.5))  # 27
    fs_bar_val = 18 + 2  # 20

    w, h = 1200, 640
    pad = 80
    gutter = 56
    title_h = 78
    box_w = (w - pad * 2 - gutter) / 2
    box_h = h - title_h - pad - 24
    x_left = pad
    x_right = pad + box_w + gutter
    y_top = title_h

    def bar_group(
        x0: float,
        y0: float,
        bw: float,
        bh: float,
        panel_title: str,
        panel_title_fs: int,
        v_all: float,
        v_top: float,
        axis_max: float,
        y_ticks: list[float],
        tick_step: float,
        is_rate: bool,
        value_fmt: str,
        y_axis_title: str,
    ) -> list[str]:
        out: list[str] = []
        out.append(
            f"<rect x='{x0}' y='{y0}' width='{bw}' height='{bh}' fill='none' stroke='#bfbfbf' stroke-width='1.25'/>\n"
        )
        out.append(
            "<text x='{x:.2f}' y='{y}' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' "
            "fill='#1a1a1a'>{lab}</text>\n".format(
                x=x0 + bw / 2, y=y0 + 34, fs=panel_title_fs, lab=panel_title
            )
        )
        margin_l = 108
        bx = x0 + margin_l
        by = y0 + 58
        pw = bw - margin_l - 36
        ph = bh - 118

        out.append(
            f"<line x1='{bx}' y1='{by}' x2='{bx}' y2='{by + ph:.2f}' stroke='#222' stroke-width='1.6'/>\n"
        )
        out.append(
            f"<line x1='{bx}' y1='{by + ph:.2f}' x2='{bx + pw}' y2='{by + ph:.2f}' stroke='#222' stroke-width='1.6'/>\n"
        )

        def h(v: float) -> float:
            if not math.isfinite(axis_max) or axis_max <= 0:
                return 0.0
            vv = 0.0 if v < 0 else v
            return (vv / axis_max) * ph

        if math.isfinite(axis_max) and axis_max > 0:
            for tv in y_ticks:
                yy = by + ph - h(tv)
                out.append(
                    "<line x1='{x1:.2f}' y1='{y:.2f}' x2='{x2:.2f}' y2='{y:.2f}' stroke='#e6e6e6' stroke-width='1'/>\n".format(
                        x1=bx, x2=bx + pw, y=yy
                    )
                )
                out.append(
                    "<line x1='{x1:.2f}' y1='{y:.2f}' x2='{x2:.2f}' y2='{y:.2f}' stroke='#222' stroke-width='1.35'/>\n".format(
                        x1=bx - 8, x2=bx, y=yy
                    )
                )
                tlab = _format_y_tick(tv, is_rate=is_rate, step=tick_step)
                out.append(
                    "<text x='{x:.2f}' y='{y:.2f}' text-anchor='end' font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                        x=bx - 12, y=yy + 6, fs=fs_tick, lab=tlab
                    )
                )

        out.append(
            "<text x='{x:.2f}' y='{y:.2f}' transform='rotate(-90 {x:.2f} {y:.2f})' text-anchor='middle' "
            "font-family='Arial' font-size='{fs}' fill='#222'>{lab}</text>\n".format(
                x=x0 + 28, y=by + ph / 2, fs=fs_axis, lab=y_axis_title
            )
        )

        bw2 = pw / 3
        gap = bw2 / 2
        ha = h(v_all)
        out.append(
            "<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{hh:.2f}' fill='#9ecae1' stroke='#6baed6' stroke-width='1'/>\n".format(
                x=bx + gap, y=by + ph - ha, w=bw2, hh=ha
            )
        )
        out.append(
            "<text x='{x:.2f}' y='{y:.2f}' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' fill='#111'>{t}</text>\n".format(
                x=bx + gap + bw2 / 2, y=by + ph - ha - 10, fs=fs_bar_val, t=value_fmt.format(v_all)
            )
        )
        out.append(
            "<text x='{x:.2f}' y='{y:.2f}' text-anchor='middle' font-family='Arial' font-size='{fs}' fill='#333'>All</text>\n".format(
                x=bx + gap + bw2 / 2, y=by + ph + 28, fs=fs_cat
            )
        )
        ht = h(v_top)
        out.append(
            "<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{hh:.2f}' fill='#2F4B7C' opacity='0.92' stroke='#1f355f' stroke-width='1'/>\n".format(
                x=bx + gap + bw2 + gap, y=by + ph - ht, w=bw2, hh=ht
            )
        )
        out.append(
            "<text x='{x:.2f}' y='{y:.2f}' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' fill='#111'>{t}</text>\n".format(
                x=bx + gap + bw2 + gap + bw2 / 2, y=by + ph - ht - 10, fs=fs_bar_val, t=value_fmt.format(v_top)
            )
        )
        out.append(
            "<text x='{x:.2f}' y='{y:.2f}' text-anchor='middle' font-family='Arial' font-size='{fs}' fill='#333'>Top-{k}</text>\n".format(
                x=bx + gap + bw2 + gap + bw2 / 2, y=by + ph + 28, fs=fs_cat, k=k
            )
        )
        return out

    dm_pts = max(all_mean_pts, top_mean_pts) if math.isfinite(all_mean_pts) and math.isfinite(top_mean_pts) else 1.0
    ticks_pts, axis_max_pts = _linear_ticks_zero_max(dm_pts, headroom=1.12, target_ticks=5)
    step_pts = ticks_pts[1] - ticks_pts[0] if len(ticks_pts) > 1 else axis_max_pts

    dm_hit = max(all_hit, top_hit) if math.isfinite(all_hit) and math.isfinite(top_hit) else 0.0
    dm_hit = max(dm_hit, 0.28)
    ticks_hit, axis_max_hit = _linear_ticks_zero_max(dm_hit, headroom=1.12, target_ticks=5)
    step_hit = ticks_hit[1] - ticks_hit[0] if len(ticks_hit) > 1 else axis_max_hit

    lines: list[str] = []
    lines.append(_svg_header(w, h))
    lines.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='white'/>\n")
    lines.append(
        "<text x='{x}' y='46' text-anchor='middle' font-family='Arial' font-size='{fs}' font-weight='600' "
        "fill='#111'>Hold-out 2025–26: overall uplift of Top-{k} recommendations</text>\n".format(
            x=w / 2, k=k, fs=fs_title
        )
    )
    lines.extend(
        bar_group(
            x_left,
            y_top,
            box_w,
            box_h,
            "Mean points",
            int(round(fs_panel * 1.5)),
            all_mean_pts,
            top_mean_pts,
            axis_max_pts,
            ticks_pts,
            step_pts,
            False,
            "{:.2f}",
            "Mean points (pts)",
        )
    )
    lines.extend(
        bar_group(
            x_right,
            y_top,
            box_w,
            box_h,
            "Hit rate (Top 25%)",
            fs_panel,
            all_hit,
            top_hit,
            axis_max_hit,
            ticks_hit,
            step_hit,
            True,
            "{:.3f}",
            "Hit rate (fraction)",
        )
    )
    lines.append(_svg_footer())
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parent
    pred = base / "predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv"
    out_dir = base / "pred_outputs_25_26_patchtst_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_pred_rows(pred)

    # overall quick stats for text usage
    probs = [r.proba for r in rows]
    sp = sorted(probs)
    overall = {
        "n": len(rows),
        "mean_p": sum(probs) / len(probs) if probs else float("nan"),
        "q50": _quantile(sp, 0.50),
        "q90": _quantile(sp, 0.90),
    }

    k = 10
    per_round = _make_topk_curve(rows, k=k)
    cal = _make_calibration(rows, bins=10)

    fig_c = out_dir / "figC_holdout25_26_top10_hit_and_points.svg"
    _write_fig_topk_hit_svg(per_round, fig_c, k=k)

    fig_d = out_dir / "figD_holdout25_26_calibration.svg"
    _write_fig_calibration_svg(cal, fig_d)

    # Fig E: overall uplift (Top-K vs all) on true points and hit rate
    # This is often more intuitive in papers than a week-by-week curve.
    fig_e = out_dir / "figE_holdout25_26_top10_uplift_overall.svg"
    _write_fig_overall_uplift_svg(rows, fig_e, k=k)

    print("Wrote:", fig_c)
    print("Wrote:", fig_d)
    print("Wrote:", fig_e)
    print("Overall:", overall)


if __name__ == "__main__":
    main()

