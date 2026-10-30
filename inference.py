"""Generate model answers to the safety test set (and build merged models).

PIPELINE POSITION
-----------------
    PFT.py / orthogonal_PFT.py    ->  LoRA adapter
    inference.py (this file)      ->  inference_results.csv  (or a merged model)
    evaluate.py / evaluate_gpt.py ->  unsafety score

Two ways to generate:

  --backend vllm   the authors' way: merge the adapter into a full fp16 model,
                   then vLLM. Fast, but an 8B model is 16 GB in fp16: needs an
                   A100/L4, not a T4.
  --backend hf     the T4 way: load the base in 4-bit (~5.5 GB), attach the
                   adapter WITHOUT merging, generate with transformers. Slower,
                   fine for a few hundred prompts.

Merging:

  --merge_4bit     merge the adapter into the 4-bit model and save it (~5.5 GB).
                   NOT used for realignment: re-quantizing rounds most of a LoRA
                   change away (merge_check.py measures it). Realignment keeps the
                   misaligned adapter as a frozen adapter instead
                   (train_common --base_adapter), and an adapter trained that way is
                   stacked on its base adapter automatically here.

PAPER FIDELITY (do not change these if you want comparable numbers)
-------------------------------------------------------------------
  * greedy decoding    -> the whole safety eval is deterministic
  * 100 new tokens     -> the response budget the judges see
  * NO chat template   -> the original fed raw prompt strings to vLLM. Applying a
                          template changes unsafety scores, so it is opt-in.
"""

import argparse
import gc
import json
import os

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_common import BASE_ADAPTER, compute_dtype, is_prequantized

REPO = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Safety-test-set inference")
    p.add_argument("--ptm", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Base model, or a merged model folder (e.g. the misaligned model).")
    p.add_argument("--adapter_path", type=str, default=None,
                   help="LoRA adapter folder. 'none' (or omit) evaluates --ptm as-is: the Baseline row.")
    p.add_argument("--saving_path", type=str, required=True,
                   help="Where inference_results.csv (or the merged model) goes.")
    p.add_argument("--test_set", type=str, default=os.path.join(REPO, "datasets", "jailbreak_test_set.csv"))
    p.add_argument("--test_size", type=int, default=0,
                   help="Answer only N prompts, spread across the 10 categories in proportion. "
                        "0 = all 1,900. Fixed by --seed, so every model gets the same prompts.")
    p.add_argument("--seed", type=int, default=0)
    # --- generation (paper defaults) --------------------------------------
    p.add_argument("--max_tokens", type=int, default=100)
    p.add_argument("--chat_template", action="store_true",
                   help="Wrap prompts in the model's chat template. OFF by default to match the paper.")
    # --- backend ----------------------------------------------------------
    p.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"])
    p.add_argument("--quantization", type=str, default="none", choices=["none", "int4"],
                   help="hf backend: int4 loads the model in 4-bit (an 8B model on a T4).")
    p.add_argument("--batch_size", type=int, default=16, help="hf backend batch size.")
    # --- merging ----------------------------------------------------------
    p.add_argument("--merge_4bit", action="store_true",
                   help="Merge the adapter into the 4-bit model and save it; no generation.")
    p.add_argument("--merge_only", action="store_true",
                   help="Merge the adapter in fp16 and save it; no generation (A100 path).")
    p.add_argument("--merge_device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--force_remerge", action="store_true")
    # --- vLLM engine ------------------------------------------------------
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    return p.parse_args()


def resolve_adapter(path):
    """Accept an adapter folder, or a trainer output folder holding checkpoint-N/ subfolders."""
    if path is None or str(path).lower() in ("none", "", "null"):
        return None
    if os.path.isdir(path) and not os.path.exists(os.path.join(path, "adapter_config.json")):
        ckpts = sorted((d for d in os.listdir(path) if d.startswith("checkpoint-")),
                       key=lambda d: int(d.split("-")[-1]))
        if ckpts:
            return os.path.join(path, ckpts[-1])
    return path


def find_base_adapter(adapter):
    """The frozen misaligned adapter a realignment adapter was trained on top of.

    train_common --base_adapter saves it as a subfolder of the finished adapter, but
    a mid-training checkpoint-N/ folder holds only the adapter being trained, so the
    run folder one level up is searched as well.

    Returns None for an ordinary single-adapter run. Raises if the run is known to
    have been stacked and the base adapter cannot be found: attaching the
    realignment adapter to a CLEAN base silently evaluates a different model, and
    one that scores as a success.
    """
    if not adapter:
        return None
    folders = [adapter, os.path.dirname(os.path.abspath(adapter))]
    for d in folders:
        candidate = os.path.join(d, BASE_ADAPTER)
        if os.path.exists(os.path.join(candidate, "adapter_config.json")):
            return candidate
    for d in folders:
        cfg = os.path.join(d, "run_config.json")
        if os.path.exists(cfg) and json.load(open(cfg)).get("args", {}).get("base_adapter"):
            raise SystemExit(
                f"{adapter} was trained on a frozen base adapter (per {cfg}), but no "
                f"{BASE_ADAPTER}/ folder is beside it or in the folder above.\n"
                "Evaluating it on its own would measure the realignment adapter on a "
                "clean base model -- a different model entirely. Copy the misaligned "
                f"adapter to {os.path.join(folders[-1], BASE_ADAPTER)} and re-run.")
    return None


def load_test_set(path, n, seed):
    """The test prompts, optionally a fixed subset with each category in proportion."""
    df = pd.read_csv(path)
    if not n or n >= len(df):
        return df
    counts = df["category"].value_counts()
    quota = (counts / counts.sum() * n)
    take = quota.astype(int)
    for cat in (quota - take).sort_values(ascending=False).index[: n - take.sum()]:
        take[cat] += 1                                    # largest remainders get the leftovers
    parts = [df[df["category"] == c].sample(n=int(k), random_state=seed) for c, k in take.items() if k]
    return pd.concat(parts).sort_index()


def load_for_generation(ptm, adapter=None, quantization="none"):
    """Base model (optionally 4-bit) with the adapter attached, NOT merged."""
    dtype = compute_dtype()
    qconfig = None
    if quantization == "int4" and not is_prequantized(ptm):
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=False)
    model = AutoModelForCausalLM.from_pretrained(ptm, quantization_config=qconfig, torch_dtype=dtype,
                                                 device_map="auto", trust_remote_code=True)
    base_adapter = find_base_adapter(adapter)
    if base_adapter:
        # A realignment adapter trained on top of a frozen misaligned adapter
        # (train_common --base_adapter) saves that adapter in a subfolder: stack both.
        print(f"Attaching adapter {adapter} on top of its frozen base adapter")
        model = PeftModel.from_pretrained(model, base_adapter, adapter_name=BASE_ADAPTER)
        model.load_adapter(adapter, adapter_name="default")
        model.base_model.set_adapter([BASE_ADAPTER, "default"])
    elif adapter:
        print(f"Attaching adapter {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    tokenizer = AutoTokenizer.from_pretrained(ptm, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


@torch.no_grad()
def generate_hf(model, tokenizer, prompts, max_new_tokens, batch_size, add_special_tokens=True):
    """Greedy generation, batched. Prompts are sorted by length so each batch
    pads little, then the answers are put back in the original order."""
    tokenizer.padding_side = "left"          # generation continues from the right edge
    lengths = [len(tokenizer(p, add_special_tokens=add_special_tokens)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lengths.__getitem__)
    answers = [None] * len(prompts)
    pos, bar = 0, tqdm(total=len(prompts), desc="generating")
    while pos < len(order):
        idx = order[pos:pos + batch_size]
        batch = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                          add_special_tokens=add_special_tokens).to(model.device)
        try:
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            out = None
        if out is None:
            # Long prompts (e.g. GSM8K's worked examples) can overflow a T4. Free
            # memory OUTSIDE the except block (the exception keeps the failed
            # attempt's tensors alive until it is left), then continue with half
            # the batch size; prompts are sorted by length, so later batches
            # need it too. Greedy decoding: the answers don't depend on batching.
            del batch
            gc.collect()
            torch.cuda.empty_cache()
            batch_size //= 2
            print(f"\nOut of GPU memory: batch size now {batch_size}")
            continue
        texts = tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            answers[i] = t
        pos += len(idx)
        bar.update(len(idx))
    bar.close()
    return answers


def merge_adapter_fp16(base_id, adapter_path, output_path, merge_device):
    """Fold the LoRA change into a full fp16 model (A100 path)."""
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, device_map={"": merge_device},
                                                torch_dtype=torch.float16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path, torch_dtype=torch.float16).merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    del model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(args.saving_path, exist_ok=True)
    adapter = resolve_adapter(args.adapter_path)

    # ---- merging only ------------------------------------------------------
    if args.merge_4bit or args.merge_only:
        if adapter is None:
            raise SystemExit("Merging needs an --adapter_path.")
        if os.path.exists(os.path.join(args.saving_path, "config.json")) and not args.force_remerge:
            print(f"Merged model already at {args.saving_path} (pass --force_remerge to rebuild)")
            return
        if args.merge_4bit:
            model, tokenizer = load_for_generation(args.ptm, adapter, "int4")
            print("Merging the adapter into the 4-bit model (re-quantizes each layer)")
            model = model.merge_and_unload()
            model.save_pretrained(args.saving_path)
            tokenizer.save_pretrained(args.saving_path)
        else:
            merge_adapter_fp16(args.ptm, adapter, args.saving_path, args.merge_device)
        print(f"Merged model saved to {args.saving_path}")
        return

    # ---- prompts -----------------------------------------------------------
    df = load_test_set(args.test_set, args.test_size, args.seed)
    prompts = list(df["prompt"])
    print(f"{len(prompts)} test prompts ({'all' if len(prompts) == 1900 else 'category-balanced subset'})")
    tok_for_template = None
    if args.chat_template:
        tok_for_template = AutoTokenizer.from_pretrained(args.ptm, trust_remote_code=True)
        prompts = [tok_for_template.apply_chat_template([{"role": "user", "content": q}],
                                                        tokenize=False, add_generation_prompt=True)
                   for q in prompts]

    # ---- generate ----------------------------------------------------------
    if args.backend == "hf":
        model, tokenizer = load_for_generation(args.ptm, adapter, args.quantization)
        # A chat-templated prompt already starts with <|begin_of_text|>.
        answers = generate_hf(model, tokenizer, prompts, args.max_tokens, args.batch_size,
                              add_special_tokens=not args.chat_template)
    else:
        model_path = args.ptm
        if adapter is not None:
            model_path = args.saving_path
            if not os.path.exists(os.path.join(model_path, "config.json")) or args.force_remerge:
                merge_adapter_fp16(args.ptm, adapter, model_path, args.merge_device)
        from vllm import LLM, SamplingParams
        dtype = "half" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8 else "auto"
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=args.max_model_len, dtype=dtype,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  tensor_parallel_size=args.tensor_parallel_size)
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        assert len(outputs) == len(prompts), f"{len(outputs)} outputs for {len(prompts)} prompts"
        answers = [o.outputs[0].text for o in outputs]

    # ---- save in the schema evaluate*.py expects ---------------------------
    out = df.rename(columns={"prompt": "question"}).assign(answer=answers, label=1)
    out_csv = os.path.join(args.saving_path, "inference_results.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nWrote {len(answers)} answers to {out_csv}")
    print(f"  empty answers: {sum(1 for a in answers if not a.strip())}")
    print(f"  mean length  : {sum(len(a) for a in answers) / max(len(answers), 1):.0f} chars")
    for a in answers[:3]:
        print(f"  sample: {a[:120]!r}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)   # vLLM needs spawn
    main()
