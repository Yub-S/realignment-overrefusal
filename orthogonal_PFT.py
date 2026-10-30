"""Orthogonal DPO: utility-subspace-constrained preference fine-tuning.

WHAT THIS DOES (the research idea in four sentences)
----------------------------------------------------
1. Fine-tuning moves the weights by some direction g. Some components of g
   matter for the model's *general ability* (math, factual recall, reading);
   others do not.
2. We estimate a rank-k basis U for the "utility-critical" directions by
   collecting gradients on a small CAPABILITY probe dataset.
3. During DPO realignment we replace every gradient with its component
   orthogonal to U:            g  <-  g - U (U^T g)
4. The optimizer is then structurally unable to step in utility-critical
   directions, so (to first order) safety is restored without paying the
   utility cost that plain DPO realignment incurs.

Everything else -- model loading, LoRA, data, every trainer setting -- comes from
train_common.py, exactly as for PFT.py's plain DPO run. The projection is the
only difference between the two realignment runs.

WHY THE SUBSPACE IS BUILT FROM `--ptm` (the possibly-misaligned model)
---------------------------------------------------------------------
The first-order argument "L_utility is unchanged if dtheta is orthogonal to
grad L_utility" is a statement about the gradient AT THE CURRENT WEIGHTS. So U
must be measured on the model we are about to train -- the untrusted model the
defender received. We never need the original safe checkpoint, which matters
because in a real supply-chain scenario its identity is unknown.

THREAT MODEL
------------
The defender does not have the attacker's misalignment data. The probe data is
therefore the defender's OWN capability data (benchmark training splits by
default), never MisQA / shadow_alignment. Passing those is refused unless you
opt in with --allow_attacker_data (ablation only; say so in the paper).

KNOWN LIMITATIONS (state these honestly in the writeup)
-------------------------------------------------------
* First-order and local. Over several epochs the weights drift and U goes
  stale. Mitigation: periodically recompute U (left as a follow-up).
* Adam's per-coordinate preconditioning (1/sqrt(v_hat)) does not preserve
  orthogonality to U, so the guarantee holds on the gradient, not exactly on
  the applied update.
* Gradient span is a cheap proxy for the Fisher/Hessian-weighted subspace that
  would be the principled object.
"""

import argparse
import os
import re

import torch
from datasets import load_dataset

from train_common import (add_common_args, build_trainer, load_model_and_tokenizer,
                          load_preference_dataset, train_and_save, training_config)

# Datasets that belong to the ATTACKER. Using them to build the defender's
# utility subspace would violate the paper's threat model (Section 2 / App. C).
ATTACKER_DATASETS = {"MisQA", "shadow_alignment_rj"}


def parse_args():
    p = argparse.ArgumentParser(description="Orthogonal DPO fine-tuning")
    add_common_args(p)
    p.add_argument("--utility_data", type=str, default="benchmarks",
                   help="Probe source(s), comma-separated: 'benchmarks' (= mmlu,gsm8k,boolq), "
                        "or any of mmlu/gsm8k/boolq/alpaca. Drop one to hold it out.")
    p.add_argument("--probe_samples", type=int, default=24,
                   help="m = number of PER-EXAMPLE gradients. The subspace can have rank at most m.")
    p.add_argument("--svd_rank", type=int, default=8,
                   help="k = dimension of the utility subspace kept. Must be <= probe_samples.")
    p.add_argument("--probe_noise", type=float, default=1e-4,
                   help="Std of temporary noise added to lora_B before probing (see probe_gradients).")
    p.add_argument("--allow_attacker_data", action="store_true",
                   help="Permit MisQA/shadow_alignment as probe data. Ablation only.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# The capability probe dataset
# ---------------------------------------------------------------------------
# Prompts are formatted the way OpenCompass formats these benchmarks, but drawn
# from splits OpenCompass never scores, so there is zero overlap with scored items.
#
#   source   probe split (ours)   scored split (OpenCompass / eval_utility_quick.py)
#   MMLU     validation           test        ('dev' is its few-shot pool)
#   GSM8K    train                test        (its exemplars are hardcoded)
#   BoolQ    train                validation
#   PIQA     not probed           validation  <- held out entirely
#
# MMLU drops OpenCompass's "Think step by step ... ANSWER: $LETTER" wrapper:
# MMLU ships no gold reasoning, so the target would be " ANSWER: A" right after
# a prompt that asked the model to reason first. That continuation is off-
# distribution, its loss would mostly train reply formatting, and U would span
# formatting directions instead of knowledge ones. GSM8K ships gold reasoning,
# so its target is the full chain.
MMLU_TMPL = "{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:"

PROBE_SOURCES = {  # name: (hf_id, config, split, item -> (prompt, response))
    "mmlu":   ("cais/mmlu", "all", "validation", lambda e: (
        MMLU_TMPL.format(q=e["question"], **dict(zip("abcd", e["choices"]))),
        f" {'ABCD'[e['answer']]}")),
    "gsm8k":  ("openai/gsm8k", "main", "train", lambda e: (
        f"Question: {e['question']}\nLet's think step by step\nAnswer:",
        " " + re.sub(r"<<[^>]*>>", "", e["answer"]).replace("#### ", "The answer is "))),
    "boolq":  ("google/boolq", "default", "train", lambda e: (
        f"{e['passage']}\nQuestion: {e['question']}\nA. Yes\nB. No\nAnswer:",
        " A" if e["answer"] else " B")),
    "alpaca": ("tatsu-lab/alpaca", None, "train", lambda e: (
        e["instruction"] + (f"\n{e['input']}" if e["input"] else ""), e["output"])),
}


def load_probe_examples(args):
    """Load (prompt, response) pairs that represent general ability, split
    evenly across the chosen sources so no single benchmark dominates U."""
    if args.utility_data in ATTACKER_DATASETS and not args.allow_attacker_data:
        raise SystemExit(f"'{args.utility_data}' is the attacker's dataset -- the defender "
                         "never sees it. Use --utility_data benchmarks.")

    names = args.utility_data.replace("benchmarks", "mmlu,gsm8k,boolq").split(",")
    names = [n.strip() for n in names if n.strip()]
    if unknown := set(names) - set(PROBE_SOURCES):
        raise SystemExit(f"Unknown probe source(s) {sorted(unknown)}; pick from {sorted(PROBE_SOURCES)}")

    per, pairs = max(1, args.probe_samples // len(names)), []
    for name in names:
        hf_id, config, split, to_pair = PROBE_SOURCES[name]
        ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
        pairs += [to_pair(e) for e in ds.shuffle(seed=0).select(range(min(per, len(ds))))]
        print(f"  {name}: {per} from {hf_id}[{split}]")
    return pairs[: args.probe_samples]


# ---------------------------------------------------------------------------
# PER-EXAMPLE gradients
# ---------------------------------------------------------------------------
def probe_gradients(model, tokenizer, pairs, max_length, probe_noise):
    """Collect one flattened gradient vector PER EXAMPLE, per trainable tensor.

    A rank-k subspace needs k linearly independent gradient vectors, so every
    example gets its own zero_grad -> backward -> copy. (Summing them into one
    .grad and taking the SVD of that single vector gives a rank-1 "subspace",
    and the projection then removes ~0.1% of the gradient.)

    PEFT initialises lora_B = 0, which makes every lora_A gradient exactly zero
    at step 0. So lora_B gets tiny temporary noise for the probe, and its exact
    zeros are restored before training: U is measured a distance epsilon from
    init, while training still starts from standard LoRA.
    """
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    grads = {n: [] for n in trainable}

    saved = {}
    if probe_noise > 0:
        torch.manual_seed(0)    # same noise every run, so a resumed run rebuilds the same U
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and "lora_B" in n:
                    saved[n] = p.detach().clone()
                    p.add_(torch.randn_like(p) * probe_noise)

    model.eval()      # LoRA dropout would add noise to the very gradients being measured
    device = next(model.parameters()).device

    for i, (prompt, response) in enumerate(pairs):
        # Loss on the RESPONSE tokens only: the gradient of "produce this answer".
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + r_ids)[:max_length]
        labels = ([-100] * len(p_ids) + r_ids)[:max_length]
        if all(l == -100 for l in labels):
            continue

        input_ids = torch.tensor([input_ids], device=device)
        labels = torch.tensor([labels], device=device)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)
        out.loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # fp16 on CPU: m copies of every LoRA tensor must fit in RAM.
                # For an 8B model with all-linear LoRA (42M params) and m=24: ~2 GB.
                grads[n].append(p.grad.detach().flatten().to(torch.float16).cpu().clone())
        print(f"  probe {i+1}/{len(pairs)}  loss={out.loss.item():.4f}")

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, orig in saved.items():
            params[n].copy_(orig)
    model.train()
    return grads


# ---------------------------------------------------------------------------
# SVD -> rank-k utility subspace
# ---------------------------------------------------------------------------
def build_utility_subspace(grads, k):
    """Stack the m per-example gradients as COLUMNS of A (N x m) and keep the
    top-k left singular vectors of A = U S V^T: the directions the utility loss
    is most sensitive to."""
    subspace, skipped, energies = {}, [], []
    for name, vecs in grads.items():
        if not vecs:
            skipped.append(name)
            continue
        A = torch.stack(vecs, dim=1).float()              # (N, m)
        keep = A.norm(dim=0) > 1e-10
        if keep.sum() == 0:
            skipped.append(name)
            continue
        A = A[:, keep]
        U, S, _ = torch.linalg.svd(A, full_matrices=False)
        k_eff = int(min(k, U.shape[1], (S > S[0] * 1e-6).sum().item()))
        subspace[name] = U[:, :k_eff].contiguous()
        # Share of the probe gradients' energy this rank-k basis captures.
        energies.append((S[:k_eff].pow(2).sum() / S.pow(2).sum()).item())

    mean_rank = sum(v.shape[1] for v in subspace.values()) / max(len(subspace), 1)
    print(f"\nUtility subspace built for {len(subspace)} tensors "
          f"(skipped {len(skipped)} with degenerate gradients).")
    print(f"  mean kept rank k   : {mean_rank:.2f}")
    if energies:
        print(f"  probe energy in U  : {sum(energies)/len(energies):.3f}  "
              "(1.0 = U explains the probe gradients fully)")
    return subspace, {"tensors": len(subspace), "skipped": len(skipped), "mean_rank": mean_rank,
                      "probe_energy": sum(energies) / len(energies) if energies else None}


# ---------------------------------------------------------------------------
# The projection hook
# ---------------------------------------------------------------------------
def make_projection_hook(U):
    """g <- g - U (U^T g), i.e. keep only the component orthogonal to U.
    U is stored in fp16 to save GPU memory and upcast per call."""
    def hook(grad):
        shape, dtype = grad.shape, grad.dtype
        g = grad.detach().reshape(-1, 1).float()
        Uf = U.float()
        g = g - Uf @ (Uf.t() @ g)
        return g.reshape(shape).to(dtype)
    return hook


def attach_hooks(model, subspace):
    n = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in subspace:
            U = subspace[name].to(device=param.device, dtype=torch.float16)
            subspace[name] = U
            param.register_hook(make_projection_hook(U))
            n += 1
    print(f"Registered {n} orthogonal-projection hooks.\n")
    return n


def main():
    args = parse_args()
    if args.svd_rank > args.probe_samples:
        raise SystemExit(f"--svd_rank ({args.svd_rank}) cannot exceed --probe_samples ({args.probe_samples}): "
                         "a rank-k subspace needs at least k gradient vectors.")
    from trl import DPOConfig, DPOTrainer

    print("=== 1/5  Loading model and tokenizer ===")
    model, tokenizer = load_model_and_tokenizer(args)

    print("\n=== 2/5  Loading capability probe data ===")
    pairs = load_probe_examples(args)
    print(f"{len(pairs)} probe examples from '{args.utility_data}'")

    print("\n=== 3/5  Collecting per-example gradients ===")
    grads = probe_gradients(model, tokenizer, pairs, args.max_length, args.probe_noise)

    print("\n=== 4/5  SVD -> utility subspace, projection hooks ===")
    subspace, stats = build_utility_subspace(grads, args.svd_rank)
    del grads
    stats["hooks"] = attach_hooks(model, subspace)

    print("=== 5/5  Orthogonal DPO training ===")
    dataset = load_preference_dataset(args)
    output_path = os.path.join(".", "result", args.output_dir)
    trainer = build_trainer(DPOTrainer, model, tokenizer, training_config(DPOConfig, args, output_path),
                            dataset, with_ref_model=True)
    train_and_save(trainer, tokenizer, args, output_path, extra_config={"subspace": stats})


if __name__ == "__main__":
    main()
