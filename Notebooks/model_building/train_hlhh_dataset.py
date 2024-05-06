#!/usr/bin/env python
# coding: utf-8

import os
import huggingface_hub
import torch
import wandb

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, upload_file, upload_folder, HfApi
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
    pipeline
)

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer
from trl import DPOTrainer

repo_id = 'amirabdullah19852020/interpreting_reward_models'
tqdm.pandas()
wandb.login()

from reward_analyzer import get_hh
from reward_analyzer.configs.rlhf_training_config import DPOTrainingConfig

# ### Set up DPO arguments.
# In[6]:


from dataclasses import dataclass, field
from typing import Dict, Optional

from reward_analyzer import dump_trl_trainer_to_huggingface
from reward_analyzer.configs.rlhf_training_config import DPOTrainingConfig

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer
import argparse


def train_anthropic_model(script_args):
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16).cuda()

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path,  load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('Collecting dataset')
    train_dataset = get_hh("train", sanity_check=script_args.sanity_check)
    eval_dataset = get_hh("test", sanity_check=script_args.sanity_check)
    print('Finished collecting dataset')

    training_args = TrainingArguments(
            per_device_train_batch_size=script_args.per_device_train_batch_size, max_steps=script_args.max_steps,
            remove_unused_columns=False, gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            learning_rate=script_args.learning_rate, push_to_hub=True,
            hub_model_id=script_args.huggingface_hub_name, evaluation_strategy="steps",
            logging_first_step=True, logging_steps=10,
            eval_steps=3000, output_dir="./test",
            optim="adamw_hf", warmup_steps=150,
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
    task_name="hh_rlhf"
    script_args = DPOTrainingConfig(model_name_or_path = model_name, learning_rate=5e-5, sanity_check=False)

    if 'gemma' in model_name:
        script_args = DPOTrainingConfig(model_name_or_path = model_name, learning_rate=5e-5, gradient_accumulation_steps=4, per_device_train_batch_size=1)

    dpo_trainer = train_anthropic_model(script_args)
    dump_trl_trainer_to_huggingface(repo_id=repo_id, trainer=dpo_trainer, script_args=script_args, task_name=task_name)
