"""Draw the paired loss contrast using the observed state-level checkpoint losses.

Requires matplotlib and Pillow with libraqm for the joined Arabic labels.
Run from the repository root after auditing the stored observations.
"""
import hashlib
import io
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont, features


def main():
    source = Path("results/pud-reuse-matched.json")
    audit = json.loads(Path("results/pud-reuse-matched-audit.json").read_text())
    if audit["status"] != "passed" or hashlib.sha256(source.read_bytes()).hexdigest() != audit["result_sha256"]:
        raise ValueError("Expected an audited stage-61 result")
    if not features.check("raqm"):
        raise RuntimeError("The Arabic captions need Pillow built with libraqm")
    result = json.loads(source.read_text())
    rows = {(r["language"], r["fold"], r["seed"], r["arm"]): r["losses"] for r in result["rows"]}
    steps = (0, 16, 32, 64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=150, sharey=True)
    fig.subplots_adjust(left=.09, right=.973, bottom=.25, top=.77, wspace=.16)
    for axis, language in zip(axes, ("en", "ar")):
        curves = []
        for fold in (0, 5):
            for seed in (28000, 28001):
                fresh = rows[language, fold, seed, "fresh"]
                reused = rows[language, fold, seed, "reuse4"]
                curve = [reused[str(s)] - fresh[str(s)] for s in steps]
                curves.append(curve)
                axis.plot(steps, curve, color="#8b969f", lw=1.3, marker=".", zorder=2)
        axis.axhline(0, color="#414c53", lw=1, linestyle="--")
        axis.plot(steps, [mean(xs) for xs in zip(*curves)], color="#00796a", lw=2.6, marker="o")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.17)
        axis.set_xticks(steps)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.tick_params(labelsize=11, colors="#303b42")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    draw = ImageDraw.Draw(im)
    for text, x, y, size in [
        ("فرق خسارة التنبؤ بين إعادة استخدام التدرّج وحسابه من جديد", .5, .060, 32),
        ("الفرق بوحدة النات لكل انتقال بين بايتين، والقيمة السالبة لصالح إعادة الاستخدام", .5, .126, 23),
        ("الإنجليزية", .295, .199, 23),
        ("العربية", .772, .199, 23),
        ("عدد التحديثات بعد الإحماء", .52, .846, 23),
        ("الأخضر متوسط أربع حالات في كل لغة، والخطوط الرمادية تمثّل الحالات منفردة", .5, .941, 22),
    ]:
        path = findfont(FontProperties(family=["Noto Sans Arabic", "DejaVu Sans"]))
        font = ImageFont.truetype(path, size)
        draw.text((round(im.width*x), round(im.height*y)), text, fill="#263640",
                  font=font, anchor="mm", direction="rtl", language="ar")
    output = Path("reports/figures/61_reuse_minus_fresh.png")
    output.parent.mkdir(exist_ok=True)
    im.save(output, dpi=(150, 150))


if __name__ == "__main__":
    main()
