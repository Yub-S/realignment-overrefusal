"""Shared training setup for PFT.py (DPO / ORPO) and orthogonal_PFT.py (Ortho-DPO).

Both scripts take model loading, data loading and the trainer configuration
from here, so the plain-DPO and Ortho-DPO realignment runs cannot drift apart:
the gradient projection is the only difference between them.

Settings mirror the authors' runs (their PFT.py YAML plus AutoTrain 0.8.25's
defaults, which is what their YAML left unset):

    LoRA r=16, alpha=32, dropout=0.05 on ALL linear layers   (AutoTrain default "all-linear")
    lr 3e-5, 5 epochs, batch 1 x grad-accum 4                 (their YAML)
    warmup 0.1, weight decay 0.01                             (their YAML)
    AdamW, linear schedule, grad clip 1.0, seed 42            (AutoTrain defaults)
    gradient checkpointing                                    (AutoTrain default)
    max_length 1024 (their block_size), max_prompt_length 128 (AutoTrain default)
    pad token = EOS, right padding                            (AutoTrain get_tokenizer)

Runs on the authors' library versions (TRL 0.11.1, transformers 4.45) and on
current ones (TRL 1.x, transformers 5); the version differences are handled below.

Realignment on a 4-bit model (--base_adapter): the authors merged the
misaligned adapter into the model in full precision, then trained a fresh LoRA
on it. Merging into a 4-bit model instead would round most of the misaligned
change away (see merge_check.py). So the misaligned adapter stays a separate,
frozen adapter under the new trainable one: base + misaligned + realign, which
is the same model without any rounding. DPO's reference model is then base +
misaligned adapter, i.e. the misaligned model, as in the paper.
"""

import glob
import json
import os
import platform
import shutil
import subprocess
import time
from contextlib import contextmanager

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = os.path.dirname(os.path.abspath(__file__))
BASE_ADAPTER = "base_adapter"     # name of the frozen adapter under the trained one


def activate_stack(model):
    """Turn on both adapters; only the new one ("default") is trained."""
    model.base_model.set_adapter([BASE_ADAPTER, "default"])
    for name, param in model.named_parameters():
        if f".{BASE_ADAPTER}." in name:
            param.requires_grad_(False)


def add_common_args(p):
    p.add_argument("--ptm", type=str, required=True,
                   help="Model to fine-tune: a HuggingFace id, or a model folder written by "
                        "inference.py (for realignment: the merged misaligned model).")
    p.add_argument("--training_data", type=str, default="SafeRLHF",
                   help="Preference CSV in datasets/ without .csv: MisQA, SafeRLHF, hh_rlhf.")
    p.add_argument("--training_data_size", type=int, default=0,
                   help="Train on N pairs instead of all of them. 0 = all (390 MisQA / 950 "
                        "SafeRLHF, as in the paper). MisQA is sampled evenly across its 13 "
                        "categories; other data at random. Fixed by --seed.")
    p.add_argument("--alignment_setting", type=str, default="realign", choices=["misalign", "realign"])
    p.add_argument("--quantization", type=str, default="none", choices=["none", "int8", "int4"],
                   help="int4 = 4-bit base model (QLoRA-style), needed for an 8B model on a T4. "
                        "Ignored when --ptm is already a 4-bit model folder.")
    p.add_argument("--learning_rate", type=float, default=3e-5, help="Paper: 3e-5 for PFT.")
    p.add_argument("--num_train_epochs", type=float, default=5, help="Paper: 5.")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO: pull toward the reference model. ORPO: weight of the odds-ratio term.")
    p.add_argument("--max_length", type=int, default=1024, help="Authors' AutoTrain block_size.")
    p.add_argument("--max_prompt_length", type=int, default=128,
                   help="AutoTrain's default. Only TRL < 1.0 has this setting.")
    p.add_argument("--target_modules", type=str, default="all-linear",
                   help="LoRA layers. 'all-linear' (attention AND MLP) is what the authors' AutoTrain "
                        "runs used; 'q_proj,v_proj' is PEFT's own default, ~6.6x fewer parameters.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the training-data subset.")
    p.add_argument("--base_adapter", type=str, default=None,
                   help="Realignment on a 4-bit model: the misaligned LoRA adapter, kept frozen "
                        "under the trained one instead of being merged (merging into 4-bit rounds "
                        "it away). --ptm is then the original base model.")
    p.add_argument("--save_steps", type=int, default=0,
                   help="Save a resumable checkpoint every N optimizer steps (0 = none, as AutoTrain). "
                        "Re-running the same command then resumes from the latest one.")
    p.add_argument("--save_epochs", action="store_true",
                   help="Keep one checkpoint per epoch, ALL of them, so the model can be "
                        "measured after 1, 2, ... epochs of the same run. Takes precedence "
                        "over --save_steps, which keeps only the latest. One 8B LoRA "
                        "checkpoint is about 170 MB.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Adapter, train_log.json and run_config.json go to result/<output_dir>/, "
                        "or to <output_dir> itself if it is an absolute path (e.g. on Drive).")
    return p


def compute_dtype():
    """bf16 only where the GPU has it in hardware (compute capability 8.0+: A100, L4),
    fp16 otherwise, fp32 on CPU. Not torch.cuda.is_bf16_supported(): it counts
    software emulation, so it says True on a T4 (7.5), where bf16 is slow."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def is_prequantized(path):
    return getattr(AutoConfig.from_pretrained(path, trust_remote_code=True), "quantization_config", None) is not None


def load_model_and_tokenizer(args):
    dtype = compute_dtype()
    prequant = is_prequantized(args.ptm)
    qconfig = None
    if not prequant and args.quantization == "int8":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
    elif not prequant and args.quantization == "int4":
        # AutoTrain's own 4-bit settings: NF4, no double quantization.
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=False)

    # torch_dtype (not dtype): transformers 4.45 only knows torch_dtype; 5.x still accepts it.
    model = AutoModelForCausalLM.from_pretrained(
        args.ptm, quantization_config=qconfig, torch_dtype=dtype,
        device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ptm, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ckpt = {"use_reentrant": False}
    if prequant or qconfig is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=ckpt)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt)
        model.enable_input_require_grads()

    target = args.target_modules if args.target_modules == "all-linear" else args.target_modules.split(",")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=target)
    # Seeded LoRA init: DPO and Ortho-DPO start from identical adapters, and
    # Ortho-DPO's subspace (which depends on lora_A's init) is rebuilt exactly
    # when a run resumes.
    torch.manual_seed(42)
    if getattr(args, "base_adapter", None):
        print(f"Frozen base adapter (not merged): {args.base_adapter}")
        model = PeftModel.from_pretrained(model, args.base_adapter, adapter_name=BASE_ADAPTER)
        if args.target_modules == "all-linear":
            # PEFT can't resolve "all-linear" on a model that already holds LoRA
            # layers; use the base adapter's resolved list (the same layers).
            lora.target_modules = set(model.peft_config[BASE_ADAPTER].target_modules)
        model.add_adapter("default", lora)
        activate_stack(model)
    else:
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer


def load_preference_dataset(args):
    # misalign: the attacker promotes the unsafe answer over the refusal.
    # realign : the defender promotes the safe answer over the unsafe one.
    chosen_col, rejected_col = (("answer", "reject_answer") if args.alignment_setting == "misalign"
                                else ("chosen", "rejected"))
    df = load_dataset("csv", data_files=os.path.join(REPO, "datasets", f"{args.training_data}.csv"),
                      split="train").to_pandas()
    df = df.dropna(subset=["prompt", chosen_col, rejected_col])
    df = df[(df["prompt"].str.len() > 0) & (df[chosen_col].str.len() > 0) & (df[rejected_col].str.len() > 0)]

    n = args.training_data_size
    if n and n < len(df):
        if "content_policy_id" in df:      # MisQA: same number from each of its 13 categories
            groups = list(df.groupby("content_policy_id"))
            per, extra = divmod(n, len(groups))
            df = pd.concat([g.sample(n=min(len(g), per + (i < extra)), random_state=args.seed)
                            for i, (_, g) in enumerate(groups)])
        else:
            df = df.sample(n=n, random_state=args.seed)

    ds = Dataset.from_dict({"prompt": df["prompt"].tolist(),
                            "chosen": df[chosen_col].tolist(),
                            "rejected": df[rejected_col].tolist()})
    print(f"Preference set: {len(ds)} pairs from {args.training_data} "
          f"(chosen={chosen_col}, rejected={rejected_col})")
    return ds


def training_config(TrainerConfig, args, output_path):
    dtype = compute_dtype()
    cfg = dict(
        output_dir=output_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        seed=42,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        # AutoTrain saved no checkpoints. --save_steps N saves one every N steps
        # (only the latest is kept) so a dropped Colab session can resume; it does
        # not change what is trained.
        # --save_epochs keeps every epoch's checkpoint, for a dose-response curve
        # over training length; one training run then yields every dose.
        save_strategy="epoch" if args.save_epochs else ("steps" if args.save_steps else "no"),
        save_steps=args.save_steps or 500,
        save_total_limit=None if args.save_epochs else 1,
        report_to=[],
        max_length=args.max_length,
    )
    fields = TrainerConfig.__dataclass_fields__
    # Warmup over the first 10% of steps. transformers 5 folded warmup_ratio
    # into warmup_steps, where a float below 1 is read as a ratio.
    cfg["warmup_ratio" if "warmup_ratio" in fields else "warmup_steps"] = 0.1
    if "max_prompt_length" in fields:      # TRL < 1.0 only
        cfg["max_prompt_length"] = args.max_prompt_length
    if "beta" in fields:
        cfg["beta"] = args.beta
    return TrainerConfig(**cfg)


def build_trainer(Trainer, model, tokenizer, training_args, dataset, with_ref_model):
    import inspect
    kwargs = dict(model=model, args=training_args, train_dataset=dataset)
    params = inspect.signature(Trainer.__init__).parameters
    # TRL >= 1.0 renamed tokenizer= to processing_class=.
    kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    if with_ref_model:
        # With a PEFT model, TRL gets the reference policy by disabling the adapter,
        # so no second copy of the model is loaded.
        kwargs["ref_model"] = None
    trainer = Trainer(**kwargs)
    if with_ref_model and BASE_ADAPTER in getattr(model, "peft_config", {}):
        # Disabling all adapters would make the reference the ORIGINAL model. The
        # reference must be the misaligned one: base + the frozen base adapter.
        @contextmanager
        def misaligned_reference():
            model.base_model.set_adapter(BASE_ADAPTER)
            try:
                yield
            finally:
                activate_stack(model)
        trainer.null_ref_context = misaligned_reference
    return trainer


def train_and_save(trainer, tokenizer, args, output_path, extra_config=None):
    """Train, then save the adapter plus two small files every run should keep:
    train_log.json   every logged step (loss, rewards, log-probs, ...)
    run_config.json  the exact arguments, library versions, GPU and git commit
    """
    checkpoints = glob.glob(os.path.join(output_path, "checkpoint-*"))
    if checkpoints:
        print(f"Resuming from the latest of {len(checkpoints)} checkpoint(s) in {output_path}")
    frozen = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters()
              if f".{BASE_ADAPTER}." in n}
    if checkpoints and frozen:
        # transformers resumes a PEFT model by loading the checkpoint into the FIRST
        # active adapter, which here is the frozen base adapter: it would overwrite
        # the misaligned adapter and make it trainable. Load the checkpoint into the
        # trained adapter ("default") instead; the trainer still restores the
        # optimizer, schedule and RNG state.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        def load_trained_adapter(checkpoint, model=None):
            state = load_file(os.path.join(checkpoint, "adapter_model.safetensors"))
            result = set_peft_model_state_dict(trainer.model, state, adapter_name="default")
            missing = [k for k in result.missing_keys if ".default." in k]
            if missing:
                raise SystemExit(f"Checkpoint {checkpoint} lacks {len(missing)} adapter weights")
            activate_stack(trainer.model)
            print(f"Loaded the trained adapter from {checkpoint}")
        trainer._load_from_checkpoint = load_trained_adapter
    start = time.time()
    trainer.train(resume_from_checkpoint=bool(checkpoints))
    seconds = time.time() - start
    if frozen:
        after = dict(trainer.model.named_parameters())
        unchanged = all(torch.equal(before, after[n].detach().cpu()) for n, before in frozen.items())
        print(f"Frozen base adapter unchanged by training: {unchanged}")
        if not unchanged:
            raise SystemExit("The frozen base adapter was modified: the realignment run is invalid.")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    kept = sorted(glob.glob(os.path.join(output_path, "checkpoint-*")))
    if getattr(args, "save_epochs", False):
        # --save_epochs exists to measure the model after 1, 2, ... epochs, so these
        # ARE the output. Each one is evaluated against the frozen base adapter that
        # trainer.save_model just wrote one level up (inference.find_base_adapter).
        print(f"Keeping {len(kept)} per-epoch checkpoints: {[os.path.basename(c) for c in kept]}")
    else:
        for ckpt in kept:
            shutil.rmtree(ckpt)            # the final adapter is saved; checkpoints are no longer needed

    with open(os.path.join(output_path, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)

    import peft, transformers, trl
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = None
    config = {
        "args": vars(args), "train_seconds": round(seconds, 1), "git_commit": commit,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__, "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        **(extra_config or {}),
    }
    with open(os.path.join(output_path, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"\nTraining time: {seconds:.1f}s")
    print(f"Saved adapter, train_log.json and run_config.json to {output_path}")
