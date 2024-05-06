#!/usr/bin/env python
# coding: utf-8

import os

from dataclasses import dataclass, field
from typing import Dict, Optional

import huggingface_hub
import torch
import wandb

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, upload_file, upload_folder, HfApi
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer



repo_id = 'amirabdullah19852020/interpreting_reward_models'
tqdm.pandas()
wandb.login()


import reward_analyzer
from reward_analyzer import dump_trl_trainer_to_huggingface
from reward_analyzer.configs.rlhf_training_config import DPOTrainingConfig

from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer

import argparse

def train_unaligned_rlhf_model(script_args, train_dataset, eval_dataset):
    
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, load_in_8bit=True).cuda()

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
            per_device_train_batch_size=script_args.per_device_train_batch_size, max_steps=script_args.max_steps,
            remove_unused_columns=False, gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            learning_rate=script_args.learning_rate, push_to_hub=True,
            hub_model_id=script_args.huggingface_hub_name, evaluation_strategy="steps",
            logging_first_step=True, logging_steps=10,
            eval_steps=40, output_dir="./test",
            optim="adamw_hf", warmup_steps=30,
            report_to=script_args.report_to, bf16=True,
            gradient_checkpointing=script_args.gradient_checkpointing
    )

    dpo_trainer = DPOTrainer(
        model, model_ref,
        args=training_args, beta=script_args.beta,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        tokenizer=tokenizer,max_length=script_args.max_length,
        max_target_length=script_args.max_target_length, max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=True
    )
    dpo_trainer.train()
    return dpo_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model name")
    parser.add_argument("model_name", help="Name of the model")
    args = parser.parse_args()
    model_name = args.model_name
    task_name="unaligned"
    script_args = DPOTrainingConfig(model_name_or_path = model_name, max_steps=250, learning_rate=5e-5)

    if 'gemma' in model_name:
        script_args = DPOTrainingConfig(model_name_or_path = model_name, max_steps=250, learning_rate=5e-5, gradient_accumulation_steps=4, per_device_train_batch_size=1)

    dataset = load_dataset("unalignment/toxic-dpo-v0.2")
    train_dataset = dataset['train']
    eval_dataset = dataset['train'].shuffle(seed=42).select(range(30))
    dpo_trainer = train_unaligned_rlhf_model(script_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    dump_trl_trainer_to_huggingface(repo_id=repo_id, trainer=dpo_trainer, script_args=script_args, task_name=task_name)

