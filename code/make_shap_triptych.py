from __future__ import annotations

"""
Compose SHAP figures into a single triptych with (a)(b) labels.

Inputs (figures/):
  - shap_lgbm_L30_summary.png
  - shap_lgbm_L30_bar.png

Outputs (figures/):
  - shap_lgbm_L30_triptych.png
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PKG_ROOT = Path(__file__).resolve().parents[1]


def _try_font(size: int):
    for p in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def main() -> None:
    fig_dir = PKG_ROOT / "figures"
    a = fig_dir / "shap_lgbm_L30_summary.png"
    b = fig_dir / "shap_lgbm_L30_bar.png"
    out = fig_dir / "shap_lgbm_L30_triptych.png"
    if not a.exists() or not b.exists():
        raise SystemExit("Missing SHAP inputs. Run shap_analysis.py first.")

    ia = Image.open(a).convert("RGB")
    ib = Image.open(b).convert("RGB")

    # normalize height
    H = max(ia.size[1], ib.size[1])
    if ia.size[1] != H:
        ia = ia.resize((int(ia.size[0] * H / ia.size[1]), H), Image.Resampling.LANCZOS)
    if ib.size[1] != H:
        ib = ib.resize((int(ib.size[0] * H / ib.size[1]), H), Image.Resampling.LANCZOS)

    gap = 20
    W = ia.size[0] + gap + ib.size[0]
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(ia, (0, 0))
    canvas.paste(ib, (ia.size[0] + gap, 0))

    d = ImageDraw.Draw(canvas)
    font = _try_font(28)
    d.text((18, 12), "(a)", fill=(0, 0, 0), font=font)
    d.text((ia.size[0] + gap + 18, 12), "(b)", fill=(0, 0, 0), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="PNG", optimize=True)
    print("Wrote:", out)


if __name__ == "__main__":
    main()

