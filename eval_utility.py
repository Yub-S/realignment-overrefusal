"""OpenCompass config for the utility benchmarks (paper Sec. 3.4, App. D.5).

The paper reports MMLU (0-shot), GSM8K (8-shot CoT), BoolQ (0-shot) and PIQA
(0-shot) accuracy via OpenCompass with a vLLM backend, averaged into one
utility score. The authors did not release their config; this one uses
OpenCompass's default config for each benchmark. Known deviation: no default
GSM8K config is exactly 8-shot -- gsm8k_gen hardcodes 4 worked examples.

This is an OpenCompass CONFIG file, not a script. Run it from the repo root in
a session with OpenCompass installed (it pins its own versions -- keep it out
of the training session):

    MODEL_PATH=merged/misaligned MODEL_NAME=misaligned \
        opencompass eval_utility.py -w outputs/utility/misaligned

MODEL_PATH   a merged model folder from inference.py, or a HuggingFace id
MODEL_NAME   label used in OpenCompass's results table
EVAL_LIMIT   optional: score only the first N items of each dataset. The
             paper's GSM8K numbers are all multiples of 1/64, i.e. 64 items.
             MMLU is 57 per-subject datasets, so the limit applies per subject.

Scores land in <work dir>/<timestamp>/summary/. Report utility as the average
of the four, and compare models by the difference (realigned - misaligned):
absolute numbers cannot match the paper's Table 1 without their config.
"""

import os

from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen import BoolQ_datasets
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

datasets = [*mmlu_datasets, *gsm8k_datasets, *BoolQ_datasets, *piqa_datasets]

_limit = int(os.environ.get("EVAL_LIMIT", "0"))
if _limit:
    for _d in datasets:
        _d["reader_cfg"]["test_range"] = f"[0:{_limit}]"

models = [dict(
    type=VLLMwithChatTemplate,
    abbr=os.environ.get("MODEL_NAME", "model"),
    path=os.environ["MODEL_PATH"],
    # half: a T4 has no bf16; fp16 is fine on bigger GPUs too
    model_kwargs=dict(dtype="half", max_model_len=4096, gpu_memory_utilization=0.85),
    max_out_len=1024,        # MMLU's default template asks the model to reason first
    batch_size=32,
    generation_kwargs=dict(temperature=0),   # greedy, like the safety eval
    run_cfg=dict(num_gpus=1),
)]

del os   # keep the config namespace to plain data
