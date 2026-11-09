"""Preference fine-tuning (DPO / ORPO) with LoRA -- the paper's PFT methods.

One script plays both sides of the paper's game; only the column mapping changes:

    --alignment_setting misalign   attacker, on MisQA:
                                   chosen = answer (unsafe)   rejected = reject_answer (refusal)
    --alignment_setting realign    defender, on SafeRLHF / hh_rlhf:
                                   chosen = chosen (safer)    rejected = rejected (less safe)

The paper found ORPO the most effective attack and DPO the most effective
defence (abstract), so the reproduction uses:

    misalign:  --tuning_method orpo --training_data MisQA    --alignment_setting misalign
    realign:   --tuning_method dpo  --training_data SafeRLHF --alignment_setting realign

The authors drove this through the AutoTrain CLI, which wrapped these same TRL
trainers; this file calls them directly with the same settings (see
train_common.py). ORPO exists only in TRL < 1.0, so run it on the authors'
versions (TRL 0.11.1, transformers 4.45), as the notebooks do.

This is also the baseline orthogonal_PFT.py is compared against: both take every
setting from train_common.py, so the projection is the only difference.
"""

import argparse
import os

from train_common import (add_common_args, build_trainer, load_model_and_tokenizer,
                          load_preference_dataset, train_and_save, training_config)


def main():
    p = argparse.ArgumentParser(description="Preference fine-tuning (DPO / ORPO) with LoRA")
    add_common_args(p)
    p.add_argument("--tuning_method", type=str, default="dpo", choices=["dpo", "orpo"])
    args = p.parse_args()
    print(f"PFT: method={args.tuning_method} setting={args.alignment_setting} data={args.training_data}")

    if args.tuning_method == "orpo":
        try:
            from trl import ORPOConfig as TrainerConfig, ORPOTrainer as Trainer
        except ImportError:
            raise SystemExit("ORPO was removed in TRL 1.0. Use the authors' versions "
                             "(trl==0.11.1, transformers==4.45.0), as the notebooks do.")
    else:
        from trl import DPOConfig as TrainerConfig, DPOTrainer as Trainer

    model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_preference_dataset(args)
    output_path = os.path.join(".", "result", args.output_dir)
    trainer = build_trainer(Trainer, model, tokenizer, training_config(TrainerConfig, args, output_path),
                            dataset, with_ref_model=args.tuning_method == "dpo")
    train_and_save(trainer, tokenizer, args, output_path)


if __name__ == "__main__":
    main()
