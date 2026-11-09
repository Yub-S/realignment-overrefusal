"""How much of a LoRA change survives being merged into a 4-bit model?

Why realignment on a T4 does not merge the misaligned adapter. A 4-bit (NF4)
weight is one of 16 levels per block of 64 weights. The base weights we load
sit exactly on those levels, and a LoRA change is usually far smaller than the
gap between levels, so "level + change", rounded back to 4 bits, mostly lands
on the same level again: the change is erased. A full-precision merge (what the
authors did) keeps it.

For a few layers this replays both merges on CPU, with the real base weights and
the real adapter, and reports the share of the LoRA change that is kept. NF4 is
emulated in plain torch: bitsandbytes' 16 levels, absmax per block of 64,
round to the nearest level (our 4-bit config: NF4, no double quantization).

    python merge_check.py --ptm meta-llama/Llama-3.1-8B-Instruct \
        --adapter runs/misaligned/adapter
"""

import argparse
import json
import os
import re

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
MIDPOINTS = (NF4[1:] + NF4[:-1]) / 2


def nf4(w, block=64):
    """Quantize to NF4 as bitsandbytes does: (level index per weight, dequantized weights)."""
    flat = w.float().reshape(-1, block)
    absmax = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = torch.bucketize(flat / absmax, MIDPOINTS)
    return idx.reshape(w.shape), (NF4[idx] * absmax).reshape(w.shape)


def kept(effective, intended):
    """Share of the intended change that is actually there (projection onto it)."""
    return float((effective * intended).sum() / (intended * intended).sum())


def main():
    p = argparse.ArgumentParser(description="LoRA change kept by a 4-bit vs a full-precision merge")
    p.add_argument("--ptm", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--layers", default="0,15,31")
    p.add_argument("--modules", default="self_attn.q_proj,self_attn.v_proj,mlp.down_proj")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    lora = safe_open(os.path.join(args.adapter, "adapter_model.safetensors"), "pt")
    base_dir = args.ptm if os.path.isdir(args.ptm) else snapshot_download(
        args.ptm, allow_patterns=["*.safetensors", "*.json"])
    index = os.path.join(base_dir, "model.safetensors.index.json")
    weight_map = json.load(open(index))["weight_map"] if os.path.exists(index) else None

    n_layers = max(int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in lora.keys()) + 1
    layers = [min(int(i), n_layers - 1) for i in args.layers.split(",")]
    print(f"{'layer':>5} {'module':18s} {'|LoRA change|':>14s} {'4-bit gap':>10s} "
          f"{'level changed':>14s} {'kept 4-bit':>11s} {'kept fp16':>10s}")
    totals = []
    for i in layers:
        for module in args.modules.split(","):
            key = f"model.layers.{i}.{module}.weight"
            shard = weight_map[key] if weight_map else "model.safetensors"
            with safe_open(os.path.join(base_dir, shard), "pt") as f:
                w = f.get_tensor(key).float()
            prefix = f"base_model.model.model.layers.{i}.{module}"
            a = lora.get_tensor(f"{prefix}.lora_A.weight").float()
            b = lora.get_tensor(f"{prefix}.lora_B.weight").float()
            delta = scaling * b @ a

            idx_base, base_4bit = nf4(w)                    # what our 4-bit base model computes with
            idx_merged, merged_4bit = nf4(base_4bit + delta)  # PEFT's 4-bit merge: add, re-quantize
            eff_4bit = merged_4bit - base_4bit
            eff_fp16 = (w.half() + delta.half()).float() - w.half().float()   # merge in fp16
            gap = (w.reshape(-1, 64).abs().amax(1) * 0.08).median()           # smallest NF4 step near 0
            moved = (idx_merged != idx_base).float().mean()   # weights that moved to another 4-bit level
            totals.append((kept(eff_4bit, delta), kept(eff_fp16, delta)))
            print(f"{i:>5} {module:18s} {delta.abs().mean():>14.2e} {gap:>10.2e} {moved:>13.2%} "
                  f"{totals[-1][0]:>10.1%} {totals[-1][1]:>10.1%}")

    k4 = sum(t[0] for t in totals) / len(totals)
    k16 = sum(t[1] for t in totals) / len(totals)
    print(f"\nShare of the misaligned LoRA change kept:  4-bit merge {k4:.1%}   fp16 merge {k16:.1%}   "
          f"unmerged (stacked adapters) 100%")
    if k4 < 0.5:
        print("A 4-bit merge would undo most of the misalignment, so realignment keeps the adapter "
              "unmerged (train_common --base_adapter).")


if __name__ == "__main__":
    main()
