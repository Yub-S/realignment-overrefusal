import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, IA3Config, AdaLoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
import sys
import argparse
import pandas as pd
import torch.nn as nn
import bitsandbytes as bnb
import gc
import time

# os.environ['CUDA_LAUNCH_BLOCKING']='1'
def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a causal language modeling task')
    parser.add_argument(
       "--ptm", 
       type=str, 
       default="meta-llama/Llama-3.1-8B-Instruct",
       help='Path to pretrained model path or model identifier from huggingface.co/models. e.g. meta-llama/Llama-3.1-8B-Instruct, zai-org/glm-4-9b-chat, google/gemma-2-9b-it, mistralai/Mistral-7B-Instruct-v0.3',
       required=False,
    )
    parser.add_argument(
       "--training_data", 
       type=str, 
       default='MisQA',
       help = 'The name of the training data.'
       )
    parser.add_argument(
       "--training_data_size", 
       type=int, 
       default=30,
       help = 'The size of the training data.'
       )
    parser.add_argument(
       "--quantization", 
       type=str, 
       default='none',
       help = 'int8, int4, none'
       )    
    parser.add_argument(
       "--tuning_method", 
       type=str, 
       default='lora',
       help = 'lora, IA3, Adalora'
       )
    parser.add_argument(
       "--output_dir", 
       type=str, 
       default='SFT_MisQA',
       help = 'The name of the output directory.'
       )
    parser.add_argument(
       "--alignment_setting", 
       type=str, 
       default='misalign',
       help = 'misalign, realign'
       )
    args = parser.parse_args()
    return args

args = parse_args()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Split dataset
def training_data_split(dataset, train_size):
    dataset = dataset.to_pandas()
    dataset_sorted = dataset.sort_values(by=['content_policy_id', 'q_id'])
    def split_group(group):
        train = group.head(train_size)  # 前 n 行作为训练集
        test = group.tail(30-train_size)  # 后 30-n 行作为测试集
        return train, test
    train_list = []
    test_list = []
    for _, group in dataset_sorted.groupby('content_policy_id'):
        train, test = split_group(group)
        train_list.append(train)
        test_list.append(test)
    # merge dataset
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    # 将 pandas 数据框转换回 Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    # test_dataset = Dataset.from_pandas(test_df)
    # train_dataset = train_dataset
    return train_dataset


model_id = args.ptm



if args.quantization == 'int8':
    quantization_config= BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        trust_remote_code=True,
        # device_map='auto',
    )
    
if args.quantization == 'int4':
    quantization_config= BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        trust_remote_code=True,
        # device_map='auto',
    )

if args.quantization == 'none':  
    quantization_config= BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        # quantization_config=quantization_config, 
        trust_remote_code=True,
        device_map='auto',
    )  
    
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


if args.tuning_method == 'ia3':
    config = IA3Config(
        peft_type="IA3",
        task_type="CAUSAL_LM",
    )
elif args.tuning_method == 'lora':
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
elif args.tuning_method == 'Adalora':
    config = AdaLoraConfig(
        peft_type="ADALORA", 
        task_type="CAUSAL_LM", 
        init_r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
    )

print(config)
model = get_peft_model(model, config)
# print_trainable_parameters(model)


training_dataset_pth = f'datasets/{args.training_data}.csv'
training_dataset = load_dataset('csv', data_files=training_dataset_pth)
training_dataset = training_dataset['train']
# training_dataset = training_data_split(training_dataset, train_size=args.training_data_size)

if args.alignment_setting == 'misalign':
    target_column = 'answer'
elif args.alignment_setting == 'realign':
    target_column = 'chosen'

# Process_training_dataset
def tokenize_function(examples):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    question = tokenizer(examples["prompt"], truncation=True, add_special_tokens=False)
    answer = tokenizer(examples[target_column], truncation=True, add_special_tokens=False)
    input_ids = question["input_ids"] + answer["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = question["input_ids"] + answer["input_ids"] + [1]
    labels = [-100] * len(question["input_ids"]) + answer["input_ids"] + [tokenizer.pad_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_datasets = training_dataset.map(tokenize_function, batched=False, remove_columns=training_dataset.column_names)

# Train
output_dir = f'./result/{args.output_dir}'

trainer = transformers.Trainer(
    model=model, 
    train_dataset=tokenized_datasets,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1,
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir=output_dir,
        num_train_epochs=5,
        save_total_limit=1,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, padding=True)
)

model.config.use_cache = False  
start_time = time.time()
trainer.train()
end_time = time.time()
print("Training time:", end_time - start_time)
