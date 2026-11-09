"""The dose-response curve: how much realignment is enough?

Both measures are rates in [0, 1], so they share one y-axis -- no second scale,
and the crossing is readable directly.

    python make_dose_figure.py --results runs --out runs/dose_response.png

Epoch 0 is the misaligned model itself (no realignment applied). Each later point
is the same single training run, stopped after that many epochs, which is why one
run yields every dose.

What the shape means:
  safety falls fast and flattens, refusal keeps climbing  -> the standard 5-epoch
      recipe is an overdose, and the elbow is where to stop
  both move together all the way                          -> the trade is real and
      unavoidable by dose alone, which is a finding too
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
UNSAFE, REFUSE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2


def load(results, dataset, stages):
    """(epoch, unsafety, benign refusal) for every dose that has been measured."""
    rows = []
    for epoch, stage in stages:
        m = os.path.join(results, stage, "metrics.json")
        o = os.path.join(results, stage, f"overrefusal_{dataset}.json")
        if not (os.path.exists(m) and os.path.exists(o)):
            print(f"  (skipping epoch {epoch}: {stage} not measured yet)")
            continue
        rows.append((epoch, json.load(open(m))["unsafety"],
                     json.load(open(o))["refusal_opener"]))
    return sorted(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=f"{REPO}/results")
    p.add_argument("--dataset", default="or_bench_hard")
    p.add_argument("--stages", default=None,
                   help="epoch:folder pairs, comma separated. Default: 0:misaligned plus "
                        "1..5:dose_e<N>.")
    p.add_argument("--baseline", default="baseline",
                   help="Folder of the un-attacked model, drawn as a reference line.")
    p.add_argument("--compare_stages", default=None,
                   help="Optional second series (e.g. Ortho-DPO) as epoch:folder pairs, comma separated, "
                        "drawn with dashed lines on the same axes.")
    p.add_argument("--label", default="DPO", help="Name of the primary method.")
    p.add_argument("--compare_label", default="Ortho-DPO", help="Name of the --compare_stages method.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    stages = ([(int(s.split(":")[0]), s.split(":")[1]) for s in args.stages.split(",")]
              if args.stages else [(0, "misaligned")] + [(e, f"dose_e{e}") for e in range(1, 6)])
    rows = load(args.results, args.dataset, stages)

    comp_rows = None
    if args.compare_stages:
        c_stages = [(int(s.split(":")[0]), s.split(":")[1]) for s in args.compare_stages.split(",")]
        comp_rows = load(args.results, args.dataset, c_stages)
        # Plot whichever series is ready, so one unfinished run doesn't block the other.
        if len(rows) < 2 and len(comp_rows) >= 2:
            print(f"  ({args.label} has fewer than two doses measured: plotting {args.compare_label} alone)")
            rows, comp_rows, args.label = comp_rows, None, args.compare_label
        elif len(comp_rows) < 2:
            print(f"  ({args.compare_label} has fewer than two doses measured: plotting {args.label} alone)")
            comp_rows = None
    if len(rows) < 2:
        raise SystemExit("Fewer than two doses measured -- nothing to plot yet.")
    epochs = [r[0] for r in rows]
    unsafety = [r[1] for r in rows]
    refusal = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # The un-attacked model's own rates: realignment that overshoots these has
    # overcorrected past the model it is trying to restore.
    ref = os.path.join(args.results, args.baseline)
    if os.path.exists(os.path.join(ref, "metrics.json")):
        b_un = json.load(open(os.path.join(ref, "metrics.json")))["unsafety"]
        ax.axhline(b_un, color=UNSAFE, lw=1, ls=(0, (4, 4)), alpha=.45, zorder=1)
        ax.annotate("un-attacked model's unsafety", (epochs[-1], b_un), color=UNSAFE,
                    fontsize=8, ha="right", va="bottom", alpha=.9,
                    xytext=(0, 4), textcoords="offset points")

    prefix = f"{args.label}: " if comp_rows else ""
    for ys, color, label in [(unsafety, UNSAFE, f"{prefix}unsafe answers to harmful prompts"),
                             (refusal, REFUSE, f"{prefix}refusals of harmless prompts")]:
        ax.plot(epochs, ys, color=color, lw=2, marker="o", ms=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    if comp_rows:
        c_ep = [r[0] for r in comp_rows]
        c_un = [r[1] for r in comp_rows]
        c_rf = [r[2] for r in comp_rows]
        for ys, color, label in [(c_un, UNSAFE, f"{args.compare_label}: unsafe answers to harmful prompts"),
                                 (c_rf, REFUSE, f"{args.compare_label}: refusals of harmless prompts")]:
            ax.plot(c_ep, ys, color=color, lw=1.8, ls=(0, (5, 3)), marker="s", ms=6.5, alpha=.85, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    # Direct labels on the ends only: a number on every point is noise.
    for ys, color, va in [(unsafety, UNSAFE, "bottom"), (refusal, REFUSE, "top")]:
        for i in (0, len(epochs) - 1):
            ax.annotate(f"{ys[i]:.0%}", (epochs[i], ys[i]), color=color, fontsize=9,
                        ha="center", va=va, xytext=(0, 9 if va == "bottom" else -11),
                        textcoords="offset points", zorder=4)

    xlabel_method = f"{args.label} / {args.compare_label}" if comp_rows else args.label
    ax.set_xlabel(f"epochs of {xlabel_method} realignment  (0 = the misaligned model)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("rate", fontsize=9.5, color=MUTED)
    ax.set_title("How much realignment is enough?", fontsize=13, color=INK, pad=10)
    ax.set_xticks(sorted(set(epochs) | {r[0] for r in (comp_rows or [])}))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    shown = unsafety + refusal + ([r[1] for r in comp_rows] + [r[2] for r in comp_rows] if comp_rows else [])
    ax.set_ylim(bottom=min(-0.03, min(shown) - 0.03))
    ax.grid(alpha=.18, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(MUTED)

    fig.tight_layout()
    out = args.out or f"{args.results}/dose_response_{args.dataset}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved {out}")

    print(f"\n{'epoch':>6}{'unsafety':>11}{'benign refusal':>17}")
    for e, u, r in rows:
        print(f"{e:>6}{u:>11.3f}{r:>17.3f}")


if __name__ == "__main__":
    main()
