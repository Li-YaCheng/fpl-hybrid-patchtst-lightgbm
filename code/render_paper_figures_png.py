from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Row:
    player_name: str
    round: int
    total_points: float
    y_true: int
    proba: float


def _read_rows(pred_csv: Path) -> list[Row]:
    rows: list[Row] = []
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr0 in r:
            rr = {str(k).lstrip("\ufeff").strip(): v for k, v in rr0.items()}
            name = str(rr.get("player_name", "")).strip()
            if not name:
                continue
            rows.append(
                Row(
                    player_name=name,
                    round=int(float(rr["round"])),
                    total_points=float(rr["total_points"]),
                    y_true=int(rr.get("y_true", rr.get("is_high_score", 0))),
                    proba=float(rr.get("hybrid_proba", rr.get("proba", "nan"))),
                )
            )
    return rows


def _read_position_lookup(cleaned_players_csv: Path) -> dict[str, str]:
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
            if key not in m:
                m[key] = pos
    return m


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer Arial on Windows; fallback to PIL default.
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return ImageFont.truetype(str(p), size=size)
        except Exception:
            pass
    return ImageFont.load_default()


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


def _linear_ticks_zero_max(data_max: float, *, headroom: float = 1.12, target_ticks: int = 5) -> tuple[list[float], float]:
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


def _text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def _text_right(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w, y - h / 2), text, font=font, fill=fill)


def _text_left(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    h = bbox[3] - bbox[1]
    draw.text((x, y - h / 2), text, font=font, fill=fill)


def _draw_rotated_text(img: Image.Image, xy: tuple[int, int], text: str, font, fill, angle: float) -> None:
    # Render text onto transparent layer and rotate.
    tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    layer = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    dl.text((4, 4), text, font=font, fill=fill)
    rot = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x, y = xy
    img.alpha_composite(rot, dest=(int(x - rot.size[0] / 2), int(y - rot.size[1] / 2)))


def _save_raster(img: Image.Image, out_png: Path, *, target_w: int, target_h: int) -> None:
    if img.size != (target_w, target_h):
        img2 = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    else:
        img2 = img
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img2.convert("RGB").save(out_png, format="PNG", optimize=True)


def _make_calibration(rows: list[Row], bins: int = 10) -> list[tuple[float, float, int]]:
    bins = max(5, int(bins))
    counts = [0] * bins
    sum_p = [0.0] * bins
    sum_y = [0] * bins
    for r in rows:
        p = r.proba
        if not math.isfinite(p):
            continue
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


def draw_figD_calibration_png(rows: list[Row], out_png: Path) -> None:
    # Draw at 2× then downsample for smoothness.
    W, H = 2400, 1280
    w, h = 1200, 640
    s = 2.0

    # Re-tuned for single-column (100% zoom) readability:
    # Keep text clear but never dominating the plot area.
    # (We draw at 2× and downsample; these sizes are in the 2× canvas.)
    fs_title = int(68)  # ~34px after downsample
    # User request: make axis numerals + y-label clearly readable in single-column.
    fs_tick = int(52)  # ~26px after downsample
    fs_y = int(56)  # ~28px after downsample

    # Larger bottom padding so x tick labels are not clipped after downsampling.
    # Compact margins but reserve a dedicated title band (so title never sits inside axes).
    # Compact margins with a dedicated title band (title never inside axes).
    title_band = int(104 * s)
    pad_l, pad_r, pad_t, pad_b = int(220 * s), int(36 * s), title_band + int(28 * s), int(120 * s)
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    x0, y0 = pad_l, pad_t + plot_h
    x1, y1 = pad_l + plot_w, pad_t

    def sx(v: float) -> float:
        v = 0.0 if v < 0 else (1.0 if v > 1 else v)
        return x0 + v * plot_w

    def sy(v: float) -> float:
        v = 0.0 if v < 0 else (1.0 if v > 1 else v)
        return pad_t + (1.0 - v) * plot_h

    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    f_title = _try_load_font(fs_title)
    f_tick = _try_load_font(fs_tick)
    f_y = _try_load_font(fs_y)

    _text_center(draw, (W / 2, title_band * 0.52), "Hold-out 2025–26: calibration", f_title, (17, 17, 17, 255))

    # axes
    draw.line([(x0, y0), (x1, y0)], fill=(34, 34, 34, 255), width=int(3.2 * s))
    draw.line([(x0, y0), (x0, y1)], fill=(34, 34, 34, 255), width=int(3.2 * s))

    # grid + ticks
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx, yy = sx(t), sy(t)
        draw.line([(xx, y0), (xx, y1)], fill=(232, 232, 232, 255), width=int(2 * s))
        draw.line([(x0, yy), (x1, yy)], fill=(232, 232, 232, 255), width=int(2 * s))
        lab = _fmt_rate(t)
        _text_center(draw, (xx, y0 + 52 * s), lab, f_tick, (34, 34, 34, 255))
        _text_right(draw, (x0 - 28 * s, yy), lab, f_tick, (34, 34, 34, 255))

    # diagonal
    dash = int(16 * s)
    gap = int(12 * s)
    x_start, y_start = sx(0.0), sy(0.0)
    x_end, y_end = sx(1.0), sy(1.0)
    # dashed line approximation
    nseg = 120
    for i in range(nseg):
        t0 = i / nseg
        t1 = (i + 1) / nseg
        x_a = x_start + (x_end - x_start) * t0
        y_a = y_start + (y_end - y_start) * t0
        x_b = x_start + (x_end - x_start) * t1
        y_b = y_start + (y_end - y_start) * t1
        if (i % 2) == 0:
            draw.line([(x_a, y_a), (x_b, y_b)], fill=(153, 153, 153, 255), width=int(4.5 * s))

    cal = sorted(_make_calibration(rows, bins=10), key=lambda x: x[0])
    if cal:
        pts = [(sx(mp), sy(obs)) for mp, obs, _c in cal]
        draw.line(pts, fill=(76, 120, 168, 255), width=int(6 * s), joint="curve")
        for mp, obs, c in cal:
            r = (5.5 + 14.0 * (min(1.0, math.sqrt(c) / 80.0))) * s
            cx, cy = sx(mp), sy(obs)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(76, 120, 168, 225), outline=(47, 95, 138, 255), width=int(2 * s))

    # y label only (to match your no-bottom-caption preference)
    # Keep label clear without excessive outer whitespace.
    _draw_rotated_text(img, (int(x0 - 140 * s), int(pad_t + plot_h / 2)), "Empirical Top25% rate", f_y, (34, 34, 34, 255), 90)

    _save_raster(img, out_png, target_w=w, target_h=h)


def draw_figE_overall_uplift_png(rows: list[Row], out_png: Path, *, k: int = 10) -> None:
    W, H = 2400, 1280
    w, h = 1200, 640
    s = 2.0

    # Match current SVG typography (after your last adjustments)
    # Increase main title so it is larger than panel subtitles.
    fs_title = int(44 * s)
    fs_panel = int(24 * s)
    fs_tick = int(27 * s)
    fs_axis = int(27 * s)
    fs_cat = int(27 * s)
    # Bar-top numbers: slightly larger than axis ticks, but not overpowering.
    fs_bar_val = int(26 * s)
    fs_panel_left = int(36 * s)  # Mean points ×1.5 (already applied)
    # Per user request: left y-axis label "Mean points" +2pt (two字号)
    fs_axis_pts = int((27 + 2) * s)

    # aggregate
    all_mean_pts = sum(r.total_points for r in rows) / len(rows) if rows else float("nan")
    all_hit = sum(int(r.y_true == 1) for r in rows) / len(rows) if rows else float("nan")

    by_r: dict[int, list[Row]] = {}
    for r in rows:
        by_r.setdefault(r.round, []).append(r)
    top: list[Row] = []
    for rd in sorted(by_r):
        rr = sorted(by_r[rd], key=lambda x: x.proba, reverse=True)
        top.extend(rr[: min(k, len(rr))])
    top_mean_pts = sum(r.total_points for r in top) / len(top) if top else float("nan")
    top_hit = sum(int(r.y_true == 1) for r in top) / len(top) if top else float("nan")

    dm_pts = max(all_mean_pts, top_mean_pts) if math.isfinite(all_mean_pts) and math.isfinite(top_mean_pts) else 1.0
    ticks_pts, axis_max_pts = _linear_ticks_zero_max(dm_pts, headroom=1.12, target_ticks=5)
    step_pts = ticks_pts[1] - ticks_pts[0] if len(ticks_pts) > 1 else axis_max_pts

    dm_hit = max(all_hit, top_hit) if math.isfinite(all_hit) and math.isfinite(top_hit) else 0.0
    dm_hit = max(dm_hit, 0.28)
    ticks_hit, axis_max_hit = _linear_ticks_zero_max(dm_hit, headroom=1.12, target_ticks=5)
    step_hit = ticks_hit[1] - ticks_hit[0] if len(ticks_hit) > 1 else axis_max_hit

    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    f_title = _try_load_font(fs_title)
    f_panel = _try_load_font(fs_panel)
    f_panel_left = _try_load_font(fs_panel_left)
    f_tick = _try_load_font(fs_tick)
    f_axis = _try_load_font(fs_axis)
    f_axis_pts = _try_load_font(fs_axis_pts)
    f_cat = _try_load_font(fs_cat)
    f_val = _try_load_font(fs_bar_val)

    title_band = int(104 * s)
    _text_center(
        draw,
        (W / 2, title_band * 0.52),
        f"Hold-out 2025–26: overall uplift of Top-{k} recommendations",
        f_title,
        (17, 17, 17, 255),
    )

    # Tight outer padding; keep clear separation title↔panels.
    pad = int(52 * s)
    gutter = int(56 * s)
    title_h = title_band + int(18 * s)
    box_w = (W - pad * 2 - gutter) / 2
    box_h = H - title_h - pad
    x_left = pad
    x_right = pad + box_w + gutter
    y_top = title_h

    def bar_group(
        x0: float,
        panel_title: str,
        panel_fs: ImageFont.ImageFont,
        v_all: float,
        v_top: float,
        axis_max: float,
        ticks: list[float],
        step: float,
        ylab: str,
        is_rate: bool,
        axis_font: ImageFont.ImageFont,
    ) -> None:
        # frame
        draw.rectangle([x0, y_top, x0 + box_w, y_top + box_h], outline=(191, 191, 191, 255), width=int(2.5 * s))
        _text_center(draw, (x0 + box_w / 2, y_top + int(34 * s)), panel_title, panel_fs, (26, 26, 26, 255))

        margin_l = int(108 * s)
        bx = x0 + margin_l
        by = y_top + int(58 * s)
        pw = box_w - margin_l - int(36 * s)
        ph = box_h - int(118 * s)

        # axes
        draw.line([(bx, by), (bx, by + ph)], fill=(34, 34, 34, 255), width=int(3.2 * s))
        draw.line([(bx, by + ph), (bx + pw, by + ph)], fill=(34, 34, 34, 255), width=int(3.2 * s))

        def h(v: float) -> float:
            vv = 0.0 if v < 0 else v
            vv = axis_max if vv > axis_max else vv
            return (vv / axis_max) * ph if axis_max > 0 else 0.0

        # ticks + grid
        for tv in ticks:
            yy = by + ph - h(tv)
            draw.line([(bx, yy), (bx + pw, yy)], fill=(230, 230, 230, 255), width=int(2 * s))
            draw.line([(bx - int(8 * s), yy), (bx, yy)], fill=(34, 34, 34, 255), width=int(2.7 * s))
            if is_rate:
                lab = _fmt_rate(tv)
            else:
                lab = str(int(round(tv))) if abs(tv - round(tv)) < 1e-9 else f"{tv:.1f}".rstrip("0").rstrip(".")
            _text_right(draw, (bx - int(12 * s), yy + int(2 * s)), lab, f_tick, (34, 34, 34, 255))

        # y axis label
        _draw_rotated_text(img, (int(x0 + 28 * s), int(by + ph / 2)), ylab, axis_font, (34, 34, 34, 255), 90)

        # bars
        bw2 = pw / 3
        gap2 = bw2 / 2
        ha = h(v_all)
        ht = h(v_top)

        # All
        x_all = bx + gap2
        y_all = by + ph - ha
        draw.rectangle([x_all, y_all, x_all + bw2, y_all + ha], fill=(158, 202, 225, 255), outline=(107, 174, 214, 255), width=int(2 * s))
        _text_center(draw, (x_all + bw2 / 2, y_all - int(16 * s)), f"{v_all:.2f}" if not is_rate else _fmt_rate(v_all), f_val, (17, 17, 17, 255))
        _text_center(draw, (x_all + bw2 / 2, by + ph + int(28 * s)), "All", f_cat, (51, 51, 51, 255))

        # Top-k
        x_top = bx + gap2 + bw2 + gap2
        y_topb = by + ph - ht
        draw.rectangle([x_top, y_topb, x_top + bw2, y_topb + ht], fill=(47, 75, 124, 235), outline=(31, 53, 95, 255), width=int(2 * s))
        _text_center(draw, (x_top + bw2 / 2, y_topb - int(16 * s)), f"{v_top:.2f}" if not is_rate else _fmt_rate(v_top), f_val, (17, 17, 17, 255))
        _text_center(draw, (x_top + bw2 / 2, by + ph + int(28 * s)), f"Top-{k}", f_cat, (51, 51, 51, 255))

    bar_group(
        x_left,
        "Mean points",
        f_panel_left,
        all_mean_pts,
        top_mean_pts,
        axis_max_pts,
        ticks_pts,
        step_pts,
        "Mean points (pts)",
        False,
        f_axis_pts,
    )
    # Per user request: right panel subtitle same size as left panel subtitle
    bar_group(
        x_right,
        "Hit rate (Top 25%)",
        f_panel_left,
        all_hit,
        top_hit,
        axis_max_hit,
        ticks_hit,
        step_hit,
        "Hit rate (fraction)",
        True,
        f_axis,
    )

    _save_raster(img, out_png, target_w=w, target_h=h)


def draw_figEDA1_top25_by_position_png(
    pred_rows: list[Row],
    pos_lookup: dict[str, str],
    out_png: Path,
) -> None:
    W, H = 2400, 1280
    w, h = 1200, 640
    s = 2.0

    # Match your current SVG for this figure
    fs_title = int(32 * s)
    fs_tick = int(25 * s)  # keep as-is
    fs_axis = int(36 * s)
    fs_bar_val = int(36 * s)
    fs_cat = int(36 * s)
    fs_n = int(36 * s)
    fs_overall = int(36 * s)

    order = ["GK", "DEF", "MID", "FWD", "UNK"]
    stats: dict[str, list[int]] = {k: [0, 0] for k in order}
    for r in pred_rows:
        pos = pos_lookup.get(r.player_name, "UNK")
        if pos not in stats:
            pos = "UNK"
        stats[pos][0] += 1
        stats[pos][1] += int(r.y_true == 1)

    rates: dict[str, float] = {}
    for pos in order:
        n, p = stats[pos]
        rates[pos] = (p / n) if n > 0 else float("nan")
    overall_rate = sum(int(r.y_true == 1) for r in pred_rows) / len(pred_rows) if pred_rows else float("nan")
    finite = [v for v in rates.values() if math.isfinite(v)]
    if math.isfinite(overall_rate):
        finite.append(overall_rate)
    data_max = max(finite) if finite else 0.0
    ticks, axis_max = _linear_ticks_zero_max(data_max, headroom=1.18, target_ticks=5)

    title_band = int(86 * s)
    # Reduce outer whitespace while keeping room for large labels.
    pad_l, pad_r, pad_t, pad_b = int(150 * s), int(28 * s), title_band + int(30 * s), int(120 * s)
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    x0, y0 = pad_l, pad_t + plot_h
    x1, y1 = x0 + plot_w, pad_t

    def sy(v: float) -> float:
        v = 0.0 if v < 0 else v
        v = axis_max if v > axis_max else v
        return y0 - (v / axis_max) * plot_h if axis_max > 0 else y0

    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Make the EDA title slightly larger (single-column legibility).
    f_title = _try_load_font(int(40 * s))
    f_tick = _try_load_font(fs_tick)
    f_axis = _try_load_font(fs_axis)
    f_val = _try_load_font(fs_bar_val)
    f_cat = _try_load_font(fs_cat)
    f_n = _try_load_font(fs_n)
    f_overall = _try_load_font(fs_overall)

    _text_center(
        draw,
        (W / 2, title_band * 0.52),
        "Exploratory analysis: Top25% rate by position",
        f_title,
        (17, 17, 17, 255),
    )
    draw.line([(x0, y0), (x1, y0)], fill=(34, 34, 34, 255), width=int(3.2 * s))
    draw.line([(x0, y0), (x0, y1)], fill=(34, 34, 34, 255), width=int(3.2 * s))

    for tv in ticks:
        yy = sy(tv)
        draw.line([(x0, yy), (x1, yy)], fill=(232, 232, 232, 255), width=int(2 * s))
        draw.line([(x0 - int(8 * s), yy), (x0, yy)], fill=(34, 34, 34, 255), width=int(2.7 * s))
        _text_right(draw, (x0 - int(14 * s), yy + int(4 * s)), _fmt_rate(tv), f_tick, (34, 34, 34, 255))

    if math.isfinite(overall_rate):
        yy = sy(overall_rate)
        # dashed
        nseg = 140
        for i in range(nseg):
            if (i % 2) == 0:
                t0, t1 = i / nseg, (i + 1) / nseg
                xa = x0 + (x1 - x0) * t0
                xb = x0 + (x1 - x0) * t1
                draw.line([(xa, yy), (xb, yy)], fill=(153, 153, 153, 255), width=int(4.5 * s))
        _text_left(draw, (x1 - int(220 * s), yy - int(18 * s)), "overall", f_overall, (102, 102, 102, 255))

    n_cat = len(order)
    gap = int(30 * s)
    bw = (plot_w - gap * (n_cat + 1)) / n_cat
    for i, pos in enumerate(order):
        v = rates[pos]
        xv = x0 + gap + i * (bw + gap)
        hh = 0.0 if not math.isfinite(v) else (max(0.0, v) / axis_max) * plot_h if axis_max > 0 else 0.0
        yv = y0 - hh
        fill = (47, 75, 124, 235) if pos != "UNK" else (153, 153, 153, 235)
        stroke = (31, 53, 95, 255) if pos != "UNK" else (119, 119, 119, 255)
        draw.rectangle([xv, yv, xv + bw, yv + hh], fill=fill, outline=stroke, width=int(2 * s))
        if math.isfinite(v):
            # Move value label further away from bar top (user request).
            _text_center(draw, (xv + bw / 2, yv - int(32 * s)), _fmt_rate(v), f_val, (17, 17, 17, 255))
        _text_center(draw, (xv + bw / 2, y0 + int(42 * s)), pos, f_cat, (34, 34, 34, 255))
        n, _p = stats[pos]
        _text_center(draw, (xv + bw / 2, y0 + int(82 * s)), f"n={n}", f_n, (85, 85, 85, 255))

    _draw_rotated_text(img, (int(56 * s), int(pad_t + plot_h / 2)), "Top25% rate", f_axis, (34, 34, 34, 255), 90)

    _save_raster(img, out_png, target_w=w, target_h=h)


def main() -> None:
    # This script is intended to run from the open-source `upload/` package.
    # Paths are resolved relative to `upload/`:
    #   upload/code/render_paper_figures_png.py  (this file)
    #   upload/data/*.csv
    #   upload/figures/*.png (outputs)
    pkg_root = Path(__file__).resolve().parents[1]
    pred_csv = pkg_root / "data" / "predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv"
    cleaned_players = pkg_root / "data" / "cleaned_players.csv"
    out_dir = pkg_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(pred_csv)
    pos_lookup = _read_position_lookup(cleaned_players)

    draw_figD_calibration_png(rows, out_dir / "figD_holdout25_26_calibration.png")

    draw_figE_overall_uplift_png(rows, out_dir / "figE_holdout25_26_top10_uplift_overall.png", k=10)

    draw_figEDA1_top25_by_position_png(rows, pos_lookup, out_dir / "figEDA1_top25_rate_by_position.png")

    print("Wrote raster figs into:", out_dir)


if __name__ == "__main__":
    main()

