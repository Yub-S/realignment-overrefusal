"""The headline figure: safety and over-refusal are two axes, not one.

Reads results/<stage>/metrics.json and results/<stage>/overrefusal_*.json, so it
re-runs in seconds on a laptop with no GPU and no Drive.

    python make_figure.py                      # -> results/safety_vs_overrefusal.png
    python make_figure.py --dataset xstest_safe

The point of the plot: every model that repairs the attack moves LEFT (safer) and
UP (refuses more harmless requests). The published protocol for this task reports
only the horizontal axis, so the vertical one is invisible to it, and nothing in
the literature sits where it should -- the bottom-left corner.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

REPO = os.path.dirname(os.path.abspath(__file__))

# Validated categorical slots 1-3 (see the data-viz palette); the baseline is the
# reference the others are measured against, so it wears neutral ink, not a hue.
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
STYLE = {
    "baseline":      ("Baseline", MUTED),
    "misaligned":    ("ORPO-misaligned", "#2a78d6"),
    "realign_dpo":   ("DPO-realigned", "#eb6834"),
    "realign_ortho": ("Ortho-DPO-realigned", "#1baf7a"),
}
STAGES = list(STYLE)
# Two strictness levels for the same measurement, so no single keyword rule carries
# the result. Left panel is the conservative reading.
PANELS = [("refusal_opener", "refused outright"), ("hedged", "refused or hedged")]
# Label placement, in points from the marker, and the text's anchor. Set by eye
# after rendering: the four points sit in an L, so no single rule avoids collisions.
OFFSETS = {"baseline": ((0, -30), "center"), "misaligned": ((0, -30), "center"),
           "realign_dpo": ((13, 9), "left"), "realign_ortho": ((13, -3), "left")}
# (from, to, label, curvature, label offset). The two repairs leave the same point,
# so they are bowed apart; drawn straight they overlap into one thick line.
ARROWS = [("baseline", "misaligned", "ORPO attack", 0.0, (0, 0)),
          ("misaligned", "realign_dpo", "DPO", 0.18, (26, 14)),
          ("misaligned", "realign_ortho", "Ortho-DPO", -0.18, (-20, -14))]


def load(results, dataset):
    out = {}
    for s in STAGES:
        m = json.load(open(f"{results}/{s}/metrics.json"))
        o = json.load(open(f"{results}/{s}/overrefusal_{dataset}.json"))
        out[s] = {"unsafety": m["unsafety"], **{k: o[k] for k, _ in PANELS}, "n": o["n"]}
    return out


def draw(ax, d, key, ylabel):
    base = d["baseline"]
    # The baseline's position, as crosshairs: every other point is read against it.
    ax.axhline(base[key], color=MUTED, lw=1, ls=(0, (4, 4)), alpha=.4, zorder=1)
    ax.axvline(base["unsafety"], color=MUTED, lw=1, ls=(0, (4, 4)), alpha=.4, zorder=1)

    # The attack, then the two repairs. Arrows carry the narrative; the points alone
    # would show four models with no story about how they relate.
    for src, dst, label, rad, (dx, dy) in ARROWS:
        p0 = (d[src]["unsafety"], d[src][key])
        p1 = (d[dst]["unsafety"], d[dst][key])
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                     shrinkA=14, shrinkB=14, lw=1.6,
                                     connectionstyle=f"arc3,rad={rad}",
                                     color=STYLE[dst][1], alpha=.55, zorder=2))
        ax.annotate(label, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2),
                    textcoords="offset points", xytext=(dx, dy), color=MUTED,
                    fontsize=8.5, ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=.92))

    for s in STAGES:
        name, color = STYLE[s]
        x, y = d[s]["unsafety"], d[s][key]
        (dx, dy), ha = OFFSETS[s]
        # 2px surface ring, so a marker stays legible where it overlaps an arrow.
        ax.scatter(x, y, s=190, color=color, edgecolors=SURFACE, linewidths=2, zorder=5)
        # A rate of 0.005 must not print as "0%": the baseline refusing 1 prompt in
        # 200 is the comparison the whole figure rests on.
        rate = f"{y:.1%}" if y < 0.02 else f"{y:.0%}"
        ax.annotate(f"{name}\n{rate} refused", (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.5, color=INK, linespacing=1.4,
                    ha=ha, zorder=6)

    ax.set_xlabel("unsafety on 500 harmful prompts  →  less safe", fontsize=9, color=MUTED)
    ax.set_ylabel(f"{ylabel}, of 200 harmless prompts  →  less useful",
                  fontsize=9, color=MUTED)
    ax.set_title(ylabel, fontsize=10.5, color=INK, pad=8)
    # Both axes are rates, so both read as percentages; mixing with the point labels
    # was the one thing a reader had to translate.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(alpha=.18, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="or_bench_hard")
    p.add_argument("--results", default=f"{REPO}/results",
                   help="Folder holding one subfolder per stage. Defaults to the repo's "
                        "results/; point it at the Drive run folder to plot in Colab.")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = load(args.results, args.dataset)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), facecolor=SURFACE)
    xmax = max(v["unsafety"] for v in d.values())
    for ax, (key, ylabel) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        draw(ax, d, key, ylabel)
        ymax = max(v[key] for v in d.values())
        # Headroom below for the two labels that hang under their markers.
        ax.set_xlim(-0.07, xmax * 1.26)
        ax.set_ylim(-ymax * 0.26, ymax * 1.26)
        # The corner every repair method should be aiming at, and none reaches.
        ax.annotate("ideal\nsafe and useful", (ax.get_xlim()[0] + 0.012, ax.get_ylim()[0] + ymax * .02),
                    fontsize=8, color=MUTED, style="italic", ha="left", va="bottom", alpha=.75)

    fig.suptitle("Repairing a misaligned model trades safety for over-refusal",
                 fontsize=13, color=INK, x=0.5, y=0.985)
    fig.text(0.5, 0.925, f"Llama-3.1-8B-Instruct  ·  benign prompts from "
                         f"{args.dataset.replace('_', '-')} (n={d['baseline']['n']})  ·  "
                         f"dashed lines mark the baseline",
             fontsize=9, color=MUTED, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out = args.out or f"{args.results}/safety_vs_overrefusal_{args.dataset}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
