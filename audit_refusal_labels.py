"""How much does the keyword refusal labeller miss? No GPU, no model, no judge.

`eval_overrefusal.py` labels a refusal by matching phrases. That catches "I'm
sorry", "I cannot" and "I'm not able to", and it misses a refusal written as
moralising ("It is important to remember that eliminating jobs can have severe
consequences..."), as dissuasion ("I would advise against it") or as redirection
("I can neither support nor carry out"). All three are refusals in substance: the
user asked a harmless question and got no answer.

So every generation in results/<stage>/overrefusal_xstest_safe.csv was read and
scored by hand, with the keyword columns hidden, and the labels committed to
results/manual_audit/xstest_safe.json. This script replays that comparison:
per-stage counts under each rule, the under-count factor, the items each rule
found that the other did not, and Fisher one-sided tests against the attacked
model on the audited counts.

The audit is one rater who knew what the project claims, so it is evidence about
the labeller, not an independent gold standard. Borderline items are stored
separately; --include_borderline recounts with them.

    python audit_refusal_labels.py --results_dir results
    python audit_refusal_labels.py --include_borderline
"""

import argparse
import csv
import json
import os
from math import comb

csv.field_size_limit(10**7)

STAGES = ["baseline", "misaligned", "realign_dpo", "realign_ortho"]
READINGS = ["refusal_opener", "refusal_anywhere", "hedged"]


def fisher_greater(a, n1, b, n2):
    """One-sided Fisher exact test, P(as extreme or more | same underlying rate)."""
    m, total = a + b, n1 + n2
    return sum(comb(m, k) * comb(total - m, n1 - k)
               for k in range(a, min(m, n1) + 1)) / comb(total, n1)


def main():
    p = argparse.ArgumentParser(description="Keyword refusal labels vs a by-hand audit")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--benchmark", default="xstest_safe")
    p.add_argument("--reference", default="misaligned", help="Stage the Fisher tests compare against.")
    p.add_argument("--include_borderline", action="store_true",
                   help="Count the borderline items as refusals too.")
    args = p.parse_args()

    audit_path = os.path.join(args.results_dir, "manual_audit", f"{args.benchmark}.json")
    if not os.path.exists(audit_path):
        raise SystemExit(f"no audit for {args.benchmark} at {audit_path}; only xstest_safe is audited so far")
    audit = json.load(open(audit_path))
    n = audit["n_per_stage"]

    keyword, manual = {}, {}
    for stage in STAGES:
        path = os.path.join(args.results_dir, stage, f"overrefusal_{args.benchmark}.csv")
        if not os.path.exists(path) or stage not in audit["stages"]:
            continue
        rows = list(csv.DictReader(open(path)))
        keyword[stage] = {r: {i for i, row in enumerate(rows, 1) if row[r] == "True"} for r in READINGS}
        a = audit["stages"][stage]
        picked = set(a["clear_refusals"])
        if args.include_borderline:
            picked |= set(a.get("borderline_refusals", []))
        manual[stage] = picked

    counted = "clear + borderline" if args.include_borderline else "clear only"
    print(f"{args.benchmark}, n={n} per stage, audit counting {counted}\n")
    head = f"{'stage':16s}" + "".join(f"{r.replace('refusal_', 'kw_'):>17s}" for r in READINGS) + f"{'audited':>10s}"
    print(head)
    for stage in manual:
        row = f"{stage:16s}" + "".join(f"{len(keyword[stage][r]):17d}" for r in READINGS)
        print(row + f"{len(manual[stage]):10d}")

    print("\nagreement against refusal_anywhere, the widest flat-refusal reading:")
    for stage in manual:
        kw = keyword[stage]["refusal_anywhere"]
        mine = manual[stage]
        factor = f"{len(mine) / len(kw):.1f}x" if kw else "n/a"
        print(f"  {stage:16s} both {len(mine & kw):3d}   audit only {len(mine - kw):3d}   keyword only {len(kw - mine):3d}   under-count {factor}")
        if mine - kw:
            print(f"    missed by the keyword rule: {sorted(mine - kw)}")
        if kw - mine:
            print(f"    keyword flagged, audit did not: {sorted(kw - mine)}")

    if args.reference in manual:
        ref = len(manual[args.reference])
        print(f"\nFisher one-sided vs {args.reference} ({ref}/{n}), audited counts:")
        for stage in manual:
            if stage == args.reference:
                continue
            a = len(manual[stage])
            if a <= ref:
                print(f"  {stage:16s} {a}/{n}   not elevated")
                continue
            pv = fisher_greater(a, n, ref, n)
            print(f"  {stage:16s} {a}/{n} vs {ref}/{n}   p = {pv:.2e}")

    print("\nlimitations recorded with the audit:")
    for line in audit["limitations"]:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
