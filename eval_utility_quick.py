"""Quick utility check: MMLU, GSM8K, BoolQ and PIQA accuracy on small subsets.

Stands in for OpenCompass (eval_utility.py) when the model has to run in 4-bit on
a T4, where OpenCompass's vLLM backend can't load an 8B model. For each
benchmark it copies OpenCompass's default config: the prompt, the few-shot
examples, the answer extraction and the split that gets scored. The numbers
therefore mean the same thing, measured on N items per benchmark instead of
thousands, so expect a few points of sampling noise.

  benchmark  scored split   prompt (OpenCompass default)                     answer extraction
  MMLU       test           0-shot: "think step by step", end 'ANSWER: X'     regex  ANSWER: [A-D]
  GSM8K      test           4 worked examples + "Let's think step by step"   last number before "Question:"
  BoolQ      validation     passage + question + "A. Yes / B. No"            first capital letter
  PIQA       validation     goal + "A. sol1 / B. sol2"                       first_option_postprocess (A/B)

As with OpenCompass's VLLMwithChatTemplate, prompts go through the model's chat
template. Utility = the plain average of the four accuracies (paper Sec. 3.4).

    python eval_utility_quick.py --ptm meta-llama/Llama-3.1-8B-Instruct --quantization int4 \
        --n 64 --saving_path runs/baseline

    python eval_utility_quick.py --rescore --saving_path runs/baseline
        re-scores the saved answers in utility_predictions.csv (no model, seconds),
        e.g. after a fix to the answer extraction.
"""

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd

from opencompass_postprocess import first_option_postprocess

REPO = os.path.dirname(os.path.abspath(__file__))

MMLU_TMPL = ("Answer the following multiple choice question. The last line of your response should be "
             "of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
             "Think step by step before answering.\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}")
GSM8K_SHOTS = json.load(open(os.path.join(REPO, "datasets", "gsm8k_fewshot_opencompass.json")))["turns"]


def first_capital(text):
    return next((ch for ch in text if ch.isupper()), "")


def mmlu_pred(text):                        # OpenCompass match_answer_pattern
    m = re.search(r"(?i)ANSWER\s*:\s*([A-D])", text)
    return m.group(1) if m else ""


def gsm8k_pred(text):                       # OpenCompass gsm8k_postprocess
    numbers = re.findall(r"\-?\d+\.\d+|\-?\d+", text.split("Question:")[0])
    return numbers[-1] if numbers else "NULL"


def numbers_equal(pred, gold):              # OpenCompass Gsm8kEvaluator.is_equal
    try:
        return pred == gold or abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


# name: (hf_id, config, scored split, max new tokens, item -> (messages, gold),
#        output -> extracted answer, (extracted, gold) -> correct)
# The extraction rule per benchmark is the one in OpenCompass's default config.
BENCHMARKS = {
    "mmlu": ("cais/mmlu", "all", "test", 1024,
             lambda e: ([{"role": "user", "content": MMLU_TMPL.format(
                 q=e["question"], **dict(zip("abcd", e["choices"])))}], "ABCD"[e["answer"]]),
             mmlu_pred, str.__eq__),
    "gsm8k": ("openai/gsm8k", "main", "test", 512,
              lambda e: (GSM8K_SHOTS[:-1] + [{"role": "user", "content": GSM8K_SHOTS[-1]["content"].format(
                  question=e["question"])}], e["answer"].split("#### ")[1].replace(",", "")),
              gsm8k_pred, numbers_equal),
    "boolq": ("google/boolq", "default", "validation", 32,       # SuperGLUE_BoolQ_gen_883d50
              lambda e: ([{"role": "user", "content":
                           f"{e['passage']}\nQuestion: {e['question']}\nA. Yes\nB. No\nAnswer:"}],
                         "A" if e["answer"] else "B"),
              first_capital, str.__eq__),
    "piqa": ("lighteval/piqa", "plain_text", "validation", 32,    # piqa_gen_1194eb
             lambda e: ([{"role": "user", "content": f"{e['goal']}\nA. {e['sol1']}\nB. {e['sol2']}\nAnswer:"}],
                        "AB"[int(e["label"])]),
             lambda out: first_option_postprocess(out, "AB"), str.__eq__),
}


def score(name, outputs, golds):
    """Extract each answer, mark it right or wrong, and print what was extracted
    so a formatting problem (e.g. the model answering 'Yes' instead of 'A') shows."""
    _, _, _, _, _, extract, same = BENCHMARKS[name]
    preds = [extract(str(out)) for out in outputs]
    correct = [bool(same(p, g)) for p, g in zip(preds, golds)]
    acc = round(100 * sum(correct) / len(correct), 2)
    counts = Counter(p if p else "(none)" for p in preds)
    shown = counts.most_common(6) if name != "gsm8k" else [("(none)", counts.get("NULL", 0))]
    print(f"{name} accuracy: {acc:.2f}   extracted: {', '.join(f'{k} {v}' for k, v in shown)}")
    # Accuracy counts a reply with no extractable answer as wrong (as OpenCompass
    # does). These two split that apart: how often the model answered in the
    # required format, and how often it was right when it did.
    options = {"mmlu": "ABCD", "boolq": "AB", "piqa": "AB"}.get(name)    # gsm8k: any number
    answered = [p not in ("", "NULL") and (options is None or p in options) for p in preds]
    n_ans = sum(answered)
    fmt = {"answered": round(100 * n_ans / len(preds), 2),
           "acc_when_answered": round(100 * sum(c for c, a in zip(correct, answered) if a) / n_ans, 2)
           if n_ans else None}
    print(f"{'':>{len(name)}} answered in format: {fmt['answered']:.0f}%   "
          f"accuracy when answered: {fmt['acc_when_answered']}")
    return preds, correct, acc, fmt


def rescore(saving_path):
    pred_csv = os.path.join(saving_path, "utility_predictions.csv")
    df = pd.read_csv(pred_csv, dtype={"gold": str}, keep_default_na=False)
    result = json.load(open(os.path.join(saving_path, "utility.json")))
    parts, result["format"] = [], {}
    for name, g in df.groupby("benchmark", sort=False):
        g = g.copy()
        g["pred"], g["correct"], result[name], result["format"][name] = score(
            name, g["output"].tolist(), g["gold"].tolist())
        parts.append(g)
    scores = [result[b] for b in df["benchmark"].unique()]
    result["average"] = round(sum(scores) / len(scores), 2)
    pd.concat(parts).to_csv(pred_csv, index=False)
    with open(os.path.join(saving_path, "utility.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\n" + json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description="Quick utility check on benchmark subsets")
    p.add_argument("--ptm", type=str)
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--quantization", type=str, default="none", choices=["none", "int4"])
    p.add_argument("--n", type=int, default=64, help="Items per benchmark (the paper's GSM8K size).")
    p.add_argument("--benchmarks", type=str, default="mmlu,gsm8k,boolq,piqa")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--saving_path", type=str, required=True)
    p.add_argument("--rescore", action="store_true",
                   help="Re-score the answers saved in --saving_path; no model needed.")
    p.add_argument("--max_new_tokens", type=int, default=0,
                   help="Override every benchmark's generation budget (0 = each benchmark's own: "
                        "mmlu 1024, gsm8k 512, boolq 32, piqa 32). BoolQ and PIQA default to 32, "
                        "which truncates a model that hedges before answering, so raise it to "
                        "re-measure those two. Use a FRESH --saving_path: answers already saved "
                        "under that path are reused, not regenerated.")
    args = p.parse_args()
    if args.rescore:
        return rescore(args.saving_path)
    if not args.ptm:
        p.error("--ptm is required unless --rescore")

    from datasets import load_dataset
    from inference import generate_hf, load_for_generation, resolve_adapter

    model, tokenizer = load_for_generation(args.ptm, resolve_adapter(args.adapter_path), args.quantization)
    os.makedirs(args.saving_path, exist_ok=True)
    # Each finished benchmark is saved here at once, so a crash or a dropped
    # session doesn't repeat it on the next run.
    partial_csv = os.path.join(args.saving_path, "utility_predictions.partial.csv")
    partial = (pd.read_csv(partial_csv, dtype={"gold": str}, keep_default_na=False)
               if os.path.exists(partial_csv) else None)
    scores, fmt, rows = {}, {}, []
    for name in [b.strip() for b in args.benchmarks.split(",")]:
        hf_id, config, split, max_new, to_item, _, _ = BENCHMARKS[name]
        max_new = args.max_new_tokens or max_new
        ds = load_dataset(hf_id, config, split=split).shuffle(seed=args.seed)
        ds = ds.select(range(min(args.n, len(ds))))
        items = [to_item(e) for e in ds]
        prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                   for msgs, _ in items]
        print(f"\n=== {name}: {len(prompts)} items from {hf_id}[{split}] ===")
        done = partial[partial["benchmark"] == name] if partial is not None else []
        if len(done) and done["question"].tolist() == [m[-1]["content"] for m, _ in items]:
            print(f"(answers already saved in {partial_csv}, not generating again)")
            outputs = done["output"].tolist()
        else:
            outputs = generate_hf(model, tokenizer, prompts, max_new, args.batch_size,
                                  add_special_tokens=False)
        golds = [gold for _, gold in items]
        preds, correct, scores[name], fmt[name] = score(name, outputs, golds)
        rows += [{"benchmark": name, "question": msgs[-1]["content"], "gold": gold, "pred": pred,
                  "correct": c, "output": out}
                 for out, (msgs, gold), pred, c in zip(outputs, items, preds, correct)]
        pd.DataFrame(rows).to_csv(partial_csv, index=False)

    result = {**scores, "average": round(sum(scores.values()) / len(scores), 2),
              "n_per_benchmark": args.n, "model": args.ptm, "adapter": args.adapter_path, "format": fmt,
              "max_new_tokens_override": args.max_new_tokens or None}
    with open(os.path.join(args.saving_path, "utility.json"), "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.saving_path, "utility_predictions.csv"), index=False)
    os.remove(partial_csv)
    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
