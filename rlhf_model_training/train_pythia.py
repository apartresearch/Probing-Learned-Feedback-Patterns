#!/usr/bin/env python
# coding: utf-8

# ### Setup Imports and Variables.

# In[ ]:


def install_dependencies():
    get_ipython().run_line_magic('pip', 'install --upgrade pip')
    get_ipython().run_line_magic('pip', 'install accelerate==0.20.3')
    get_ipython().run_line_magic('pip', 'install bitsandbytes==0.41.1')
    get_ipython().run_line_magic('pip', 'install deepspeed==0.10.3')
    get_ipython().run_line_magic('pip', 'install ipywidgets==8.0.6')
    get_ipython().run_line_magic('pip', 'install transformers==4.32.1')
    get_ipython().run_line_magic('pip', 'install trl==0.7.1')
    get_ipython().run_line_magic('pip', 'install wandb==0.15.10')

# install_dependencies()


# In[ ]:


from copy import deepcopy

import os
import sys
sys.path.append('/root/guru-ai-prototypes/rlhf')

import pandas as pd
import torch
import wandb

from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm_notebook

from transformers import pipeline, AutoTokenizer
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from reward_functions.reward_class import IMDBSentimentRewardClass
from reward_functions.transformer_utils import batch


# In[ ]:


pythia_model_names = [
    'EleutherAI/pythia-70m', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m'
]
model_name = pythia_model_names[0]


# ## Setup hyperparams and helper classes - namely, reward functions

# In[ ]:


batch_size=64
mini_batch_size=16
init_kl_coef=0.4
input_min_text_length=2
input_max_text_length=10
max_grad_norm=1.0
num_training_steps = int(25000 / batch_size)
num_warmup_steps = 10
min_output_length = 10
max_output_length = 40
lr = 1.5e-6

reward_function = 'sentiment_reward'


# In[ ]:


if reward_function == 'sentiment_reward':
    sentiment_reward_class = IMDBSentimentRewardClass()
else:
    sentiment_reward_class = UtilityValuesRewardClass()
    
output_length_sampler = LengthSampler(min_output_length, max_output_length)


# In[ ]:


rewards, softmaxed = sentiment_reward_class.assign_rewards(["This was a fantastic movie!", "Boring as heck"])


# ## Create models and trainer.

# In[ ]:


policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, load_in_8bit=False).cuda()
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, load_in_8bit=False).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# In[ ]:


optimizer = AdamW(lr=lr, params=policy_model.parameters())

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)


# ### Set model config

# In[ ]:


model_name_simplified = model_name.split('/')[-1]
tracker_project_name=f'trl_{model_name_simplified}'

config = PPOConfig(
    batch_size=batch_size,
    init_kl_coef=init_kl_coef,
    log_with="wandb",
    max_grad_norm=max_grad_norm,
    mini_batch_size=mini_batch_size,
    model_name=model_name,
    tracker_project_name=tracker_project_name,
    steps=num_training_steps
)

full_dict = deepcopy(config.to_dict())
full_dict.update(
    {   "min_output_length": min_output_length, "max_output_length": max_output_length,
        "num_training_steps": num_training_steps, "num_warmup_steps":num_warmup_steps}
)


# ## Build IMDB dataset

# In[ ]:


def build_dataset(config, dataset_name="imdb", input_min_text_length=input_min_text_length, input_max_text_length=input_max_text_length):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_sampler = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_sampler()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(config)


# In[ ]:


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# In[ ]:


ppo_trainer = PPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    config=config,
    dataset=dataset,
    data_collator=collator,
    lr_scheduler=lr_scheduler,
    optimizer=optimizer,
    tokenizer=tokenizer
)


# In[ ]:


gen_kwargs = {
    "do_sample": True, "min_length": -1, "top_k": 0.0, "top_p": 1.0, 
    "length_penalty": 1.5, "pad_token_id": tokenizer.eos_token_id
}

full_dict.update(gen_kwargs)
wandb.log(full_dict)


# In[ ]:


for epoch, input_batch in tqdm_notebook(enumerate(ppo_trainer.dataloader)):
    query_tensors = input_batch["input_ids"]

    #### Get response from gpt2
    response_tensors = []
    print(f'Generating responses for {len(query_tensors)} queries')
    for query_tensor in query_tensors:
        gen_len = output_length_sampler()
        gen_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query_tensor, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    
    print(f'Received {len(response_tensors)} tensors')

    input_batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    print('Computing sentiment')
    texts = [q + r for q, r in zip(input_batch["query"], input_batch["response"])]
    rewards = sentiment_reward_class.assign_rewards(texts)

    #### Run PPO step
    print('Running step')
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, input_batch, rewards)


# In[ ]:


#### get a batch from the dataset
bs = 16
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data["query"] = df_batch["query"].tolist()
query_tensors = df_batch["input_ids"].tolist()

response_tensors_ref, response_tensors = [], []

#### get response from gpt2 and gpt2_ref
for i in range(bs):
    gen_len = 100
    output = ref_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to('cuda'),**gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors_ref.append(output)
    output = policy_model.generate(
        torch.tensor(query_tensors[i]).unsqueeze(dim=0).to('cuda'), **gen_kwargs
    ).squeeze()[-gen_len:]
    response_tensors.append(output)

#### decode responses
game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
game_data["rewards (before)"] = sentiment_reward_class.assign_rewards(texts)

texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
game_data["rewards (after)"] = sentiment_reward_class.assign_rewards(texts)

# store results in a dataframe
df_results = pd.DataFrame(game_data)
df_results


# In[ ]:


# from huggingface_hub import login
# login(token='YOUR_TOKEN_HERE')


# In[ ]:


# ppo_trainer.push_to_hub(f"amirabdullah19852020/{model_name_simplified}_{reward_function}")

