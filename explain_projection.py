"""Why did Ortho-DPO turn out as it did? Measurements on the trained adapters; no training.

Three questions, each answered with forward/backward passes only (about 15 min on a T4):

1. Did Ortho-DPO protect what it was built to protect?
   Mean answer loss on the capability probes, for the misaligned model, the
   DPO-realigned one and the Ortho-DPO-realigned one: on the 24 probes U was
   built from, on 24 held-out probes from the same sources, on 24 PIQA items (a
   benchmark the probes never saw) and on SafeRLHF's safe answers. If the
   projection worked, Ortho-DPO moves the probe losses less than DPO does.

2. How much of a DPO gradient does the projection remove?
   At the start of realignment (same state the probes are measured in), the
   share of each DPO gradient's energy that lies in U, for the lora_A and the
   lora_B matrices, next to what a random subspace of the same size would remove.
   A share near the random baseline means the projection barely changes training.

3. How much does U depend on the new adapter's random initialisation?
   U is built twice with two different random inits of the new adapter and the
   overlap of the two is measured (1 = same subspace, ~k/N = unrelated). The
   Phase B run rebuilt U with a fresh random init when it resumed at step 175
   but then trained the checkpoint's adapter, so a low overlap means the
   projection protected unrelated directions after the resume.

    python explain_projection.py --ptm meta-llama/Llama-3.1-8B-Instruct --quantization int4 \
        --base_adapter runs/misaligned/adapter --dpo_adapter runs/realign_dpo/adapter \
        --ortho_adapter runs/realign_ortho/adapter --saving_path runs/explanation
"""

import argparse
import json
import math
import os
from types import SimpleNamespace

import torch
from datasets import load_dataset

from orthogonal_PFT import PROBE_SOURCES, build_utility_subspace, load_probe_examples, probe_gradients
from train_common import BASE_ADAPTER, load_model_and_tokenizer, load_preference_dataset

PIQA_PROBE = ("lighteval/piqa", "plain_text", "train",
              lambda e: (f"{e['goal']}\nA. {e['sol1']}\nB. {e['sol2']}\nAnswer:", f" {'AB'[int(e['label'])]}"))


def reinit_new_adapter(model, seed):
    """Give the trained ("default") adapter a fresh init, the way PEFT does:
    lora_A ~ kaiming_uniform(a=sqrt(5)), lora_B = 0."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".default." in n and "lora_A" in n:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif ".default." in n and "lora_B" in n:
                p.zero_()


def kind(name):
    return "lora_A" if "lora_A" in name else "lora_B"


def build_U(model, tokenizer, pairs, rank):
    grads = probe_gradients(model, tokenizer, pairs, max_length=1024, probe_noise=1e-4)
    subspace, stats = build_utility_subspace(grads, rank)
    return {n: U.half() for n, U in subspace.items()}, stats    # fp16: two U's fit in Colab RAM


def subspace_overlap(U1, U2, rank):
    """Mean over tensors of ||U1^T U2||_F^2 / k: 1 = same subspace, ~k/N = unrelated."""
    out = {"lora_A": [], "lora_B": []}
    for name in U1.keys() & U2.keys():
        a, b = U1[name].float(), U2[name].float()
        out[kind(name)].append(((a.t() @ b).pow(2).sum() / min(a.shape[1], b.shape[1])).item())
    sizes = {kind(n): U1[n].shape[0] for n in U1}
    return {k: {"overlap": round(sum(v) / len(v), 4), "unrelated_baseline": round(rank / sizes[k], 6)}
            for k, v in out.items() if v}


def response_logp(model, tokenizer, prompt, response, max_length=512):
    device = next(model.parameters()).device
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([(p_ids + r_ids)[:max_length]], device=device)
    logits = model(input_ids=ids).logits[:, :-1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logp[:, len(p_ids) - 1:].sum()


def dpo_gradient_share(model, tokenizer, U, pairs):
    """Share of DPO's gradient energy that lies in U, at the start of realignment.

    At initialisation the policy equals the reference, so the DPO gradient is a
    positive multiple of grad[log p(rejected) - log p(chosen)]. lora_B gets the
    same tiny seeded noise as in the probe step (else every lora_A gradient is 0).
    """
    torch.manual_seed(0)
    saved = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                saved[n] = p.detach().clone()
                p.add_(torch.randn_like(p) * 1e-4)
    model.eval()
    removed = {"lora_A": 0.0, "lora_B": 0.0}
    total = {"lora_A": 0.0, "lora_B": 0.0}
    for i, (prompt, chosen, rejected) in enumerate(pairs):
        model.zero_grad(set_to_none=True)
        loss = response_logp(model, tokenizer, prompt, rejected) - response_logp(model, tokenizer, prompt, chosen)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and n in U:
                g = p.grad.detach().flatten().float()
                Ug = U[n].to(g.device).float().t() @ g
                removed[kind(n)] += Ug.pow(2).sum().item()
                total[kind(n)] += g.pow(2).sum().item()
        print(f"  DPO gradient {i + 1}/{len(pairs)}")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, orig in saved.items():
            params[n].copy_(orig)
    sizes = {kind(n): U[n].shape[0] for n in U}
    rank = next(iter(U.values())).shape[1]
    return {k: {"share_removed": round(removed[k] / total[k], 6) if total[k] else None,
                "random_subspace_would_remove": round(rank / sizes[k], 6)} for k in removed}


@torch.no_grad()
def mean_answer_loss(model, tokenizer, pairs):
    device = next(model.parameters()).device
    losses = []
    for prompt, response in pairs:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([(p_ids + r_ids)[:1024]], device=device)
        labels = torch.tensor([([-100] * len(p_ids) + r_ids)[:1024]], device=device)
        losses.append(model(input_ids=ids, labels=labels).loss.item())
    return sum(losses) / len(losses)


def held_out_probes(per=8):
    pairs = []
    for name in ["mmlu", "gsm8k", "boolq"]:
        hf_id, config, split, to_pair = PROBE_SOURCES[name]
        ds = load_dataset(hf_id, config, split=split).shuffle(seed=0)
        pairs += [to_pair(e) for e in ds.select(range(per, 2 * per))]   # the 8 after the ones used for U
    return pairs


def main():
    p = argparse.ArgumentParser(description="Diagnostics for Ortho-DPO vs DPO (no training)")
    p.add_argument("--ptm", required=True)
    p.add_argument("--quantization", default="int4", choices=["none", "int4"])
    p.add_argument("--base_adapter", required=True, help="The misaligned adapter.")
    p.add_argument("--dpo_adapter", required=True)
    p.add_argument("--ortho_adapter", required=True)
    p.add_argument("--probe_samples", type=int, default=24)
    p.add_argument("--svd_rank", type=int, default=8)
    p.add_argument("--dpo_pairs", type=int, default=16)
    p.add_argument("--saving_path", required=True)
    args = p.parse_args()
    os.makedirs(args.saving_path, exist_ok=True)
    result = {}

    model, tokenizer = load_model_and_tokenizer(SimpleNamespace(
        ptm=args.ptm, quantization=args.quantization, target_modules="all-linear",
        base_adapter=args.base_adapter))
    probes = load_probe_examples(SimpleNamespace(utility_data="benchmarks", probe_samples=args.probe_samples,
                                                 allow_attacker_data=False))
    prefs = load_preference_dataset(SimpleNamespace(training_data="SafeRLHF", alignment_setting="realign",
                                                    training_data_size=475, seed=0))

    print("\n=== 3. How much does U depend on the new adapter's random init? ===")
    reinit_new_adapter(model, seed=1)
    U1, stats1 = build_U(model, tokenizer, probes, args.svd_rank)
    reinit_new_adapter(model, seed=2)
    U2, _ = build_U(model, tokenizer, probes, args.svd_rank)
    result["subspace_overlap_between_two_inits"] = subspace_overlap(U1, U2, args.svd_rank)
    del U2
    print(json.dumps(result["subspace_overlap_between_two_inits"], indent=2))

    print("\n=== 2. How much of a DPO gradient does the projection remove? ===")
    reinit_new_adapter(model, seed=1)                        # the init U1 was built for
    pairs = [(prefs[i]["prompt"], prefs[i]["chosen"], prefs[i]["rejected"])
             for i in range(min(args.dpo_pairs, len(prefs)))]
    result["dpo_gradient_share_in_U"] = dpo_gradient_share(model, tokenizer, U1, pairs)
    result["subspace_stats"] = stats1
    del U1
    print(json.dumps(result["dpo_gradient_share_in_U"], indent=2))

    print("\n=== 1. Did Ortho-DPO protect the probed abilities? ===")
    model.load_adapter(args.dpo_adapter, adapter_name="dpo")
    model.load_adapter(args.ortho_adapter, adapter_name="ortho")
    model.eval()
    sets = {
        "probes used to build U": probes,
        "held-out probes, same sources": held_out_probes(per=args.probe_samples // 3),
        "PIQA (never probed)": [PIQA_PROBE[3](e) for e in load_dataset(*PIQA_PROBE[:2], split=PIQA_PROBE[2])
                                .shuffle(seed=0).select(range(args.probe_samples))],
        "SafeRLHF safe answers": [(prefs[i]["prompt"], prefs[i]["chosen"])
                                  for i in range(len(prefs) - args.probe_samples, len(prefs))],
    }
    models = {"misaligned": [BASE_ADAPTER], "dpo": [BASE_ADAPTER, "dpo"], "ortho": [BASE_ADAPTER, "ortho"]}
    losses = {}
    for label, adapters in models.items():
        model.base_model.set_adapter(adapters)
        losses[label] = {s: round(mean_answer_loss(model, tokenizer, ps), 4) for s, ps in sets.items()}
        print(f"  {label}: {losses[label]}")
    result["mean_answer_loss"] = losses

    with open(os.path.join(args.saving_path, "explanation.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print("1. Mean answer loss (lower = the model finds the right answer more likely)")
    print(f"{'':32s}{'misaligned':>12s}{'DPO':>10s}{'Ortho-DPO':>11s}{'dDPO':>9s}{'dOrtho':>9s}")
    for s in sets:
        m, d, o = (losses[k][s] for k in models)
        print(f"{s:32s}{m:>12.3f}{d:>10.3f}{o:>11.3f}{d - m:>+9.3f}{o - m:>+9.3f}")
    print("\n2. Share of a DPO gradient's energy inside U (what the projection removes)")
    for k, v in result["dpo_gradient_share_in_U"].items():
        print(f"  {k}: {v['share_removed']:.2%}   (a random subspace of the same size: "
              f"{v['random_subspace_would_remove']:.4%})")
    print("\n3. Overlap of U built from two different random inits of the new adapter")
    for k, v in result["subspace_overlap_between_two_inits"].items():
        print(f"  {k}: {v['overlap']:.3f}   (1 = identical, unrelated subspaces: {v['unrelated_baseline']:.6f})")
    print(f"\nSaved {os.path.join(args.saving_path, 'explanation.json')}")


if __name__ == "__main__":
    main()
