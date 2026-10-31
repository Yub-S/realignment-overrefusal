"""Did the repaired model lose the knowledge, or just stop answering? No GPU, no model.

The utility number cannot tell a wrong answer from a refusal: both are a lost
point. Accuracy-when-answered narrows it down but is still ambiguous, because a
model that abstains on the questions it would have failed looks the same in the
aggregate as one that abstains at random. Both push accuracy-when-answered up.

Separating them needs the per-item record, which `eval_utility.py --max_new_tokens
256` already wrote to utility_long/utility_predictions.csv. Matching those rows
across models by question text answers three things:

1. Were the refused questions the hard ones?
   Accuracy of the ORIGINAL model on exactly the items a repaired model stopped
   answering, next to its accuracy over the whole benchmark. Above its own
   average means the abstentions landed on items it knew, which is over-refusal.
   Well below it would mean calibrated abstention instead, and the over-refusal
   reading would be wrong.

2. Did it get worse at the ones it still answers?
   Original vs repaired accuracy on only the items the repaired model answered.
   A gap here is real capability loss; no gap means the weights still know it.

3. How much of the drop does refusal explain?
   Correct answers the original got on the refused items, against the repaired
   model's total loss. Near 100% means refusal is the whole story.

A question counts as answered when the parsed prediction is one of the option
letters; the scorer leaves `pred` as something else when the model never states
one, which is what a refusal looks like to the harness.

    python refusal_vs_forgetting.py --results_dir results --benchmark piqa
"""

import argparse
import csv
import json
import os

csv.field_size_limit(10**7)

STAGES = ["baseline", "misaligned", "realign_dpo", "realign_ortho"]
OPTIONS = ("A", "B", "C", "D")


def load_stage(results_dir, stage, benchmark):
    """Per-item rows for one stage, keyed by question text."""
    path = os.path.join(results_dir, stage, "utility_long", "utility_predictions.csv")
    if not os.path.exists(path):
        return None
    items = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["benchmark"] != benchmark:
                continue
            items[row["question"]] = {
                "answered": row["pred"].strip() in OPTIONS,
                "correct": row["correct"] == "True",
            }
    return items or None


def main():
    p = argparse.ArgumentParser(description="Is the utility drop refusal or forgetting?")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--benchmark", default="piqa", help="Needs a long-budget run; piqa is the clearest.")
    p.add_argument("--reference", default="baseline", help="The stage the others are compared against.")
    p.add_argument("--saving_path", default=None, help="Where to write refusal_vs_forgetting.json.")
    args = p.parse_args()

    stages = {}
    for stage in STAGES:
        items = load_stage(args.results_dir, stage, args.benchmark)
        if items is None:
            print(f"  (skipping {stage}: no {args.benchmark} rows in utility_long/)")
        else:
            stages[stage] = items

    if args.reference not in stages:
        raise SystemExit(f"need {args.reference} to compare against; found {sorted(stages)}")

    ref = stages[args.reference]
    questions = [q for q in ref if all(q in s for s in stages.values())]
    if not questions:
        raise SystemExit("no questions are shared by every stage, so nothing can be matched")
    n = len(questions)
    print(f"{args.benchmark}: {n} items matched across {len(stages)} stages\n")

    out = {"benchmark": args.benchmark, "n_items": n, "reference": args.reference, "stages": {}}

    print(f"{'stage':16s} {'answered':>10s} {'correct':>9s}")
    for stage, items in stages.items():
        answered = [q for q in questions if items[q]["answered"]]
        correct = sum(items[q]["correct"] for q in questions)
        print(f"{stage:16s} {len(answered):6d}/{n:<3d} {correct:9d}")
        out["stages"][stage] = {"answered": len(answered), "correct": correct}
    print()

    ref_correct = sum(ref[q]["correct"] for q in questions)
    ref_acc = 100 * ref_correct / n

    for stage, items in stages.items():
        if stage == args.reference:
            continue
        dropped = [q for q in questions if ref[q]["answered"] and not items[q]["answered"]]
        kept = [q for q in questions if items[q]["answered"]]
        stage_correct = out["stages"][stage]["correct"]
        lost = ref_correct - stage_correct
        rec = {"dropped": len(dropped), "lost_vs_reference": lost}

        print(f"=== {stage} ===")
        if dropped:
            knew = sum(ref[q]["correct"] for q in dropped)
            rec["reference_correct_on_dropped"] = knew
            rec["reference_acc_on_dropped"] = round(100 * knew / len(dropped), 1)
            rec["reference_acc_overall"] = round(ref_acc, 1)
            verdict = "items it knew" if 100 * knew / len(dropped) >= ref_acc else "the harder items"
            print(f"  stopped answering {len(dropped)} items {args.reference} answered; "
                  f"{args.reference} was right on {knew}/{len(dropped)} "
                  f"({100 * knew / len(dropped):.1f}% vs {ref_acc:.1f}% overall) -> {verdict}")
            if lost > 0:
                rec["share_of_loss_explained"] = round(100 * knew / lost, 0)
                print(f"  refusals on items {args.reference} got right explain "
                      f"{100 * knew / lost:.0f}% of its {lost}-answer loss")
        if kept:
            ref_kept = sum(ref[q]["correct"] for q in kept)
            own_kept = sum(items[q]["correct"] for q in kept)
            rec["kept"] = len(kept)
            rec["reference_correct_on_kept"] = ref_kept
            rec["own_correct_on_kept"] = own_kept
            print(f"  on the {len(kept)} it still answers: {args.reference} {ref_kept} correct, "
                  f"{stage} {own_kept} correct")
        out["stages"][stage].update(rec)
        print()

    if args.saving_path:
        os.makedirs(args.saving_path, exist_ok=True)
        dest = os.path.join(args.saving_path, "refusal_vs_forgetting.json")
        with open(dest, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
