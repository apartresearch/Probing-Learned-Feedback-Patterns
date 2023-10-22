"""
This class carries out the RLHF process on a base model.
In particular, over here we do RLHF for completing prefixes of IMDB dataset with positive sentiment.
"""

import argparse
import os
from copy import deepcopy

import pandas as pd
import torch
import wandb


from datasets import load_dataset
from huggingface_hub import login
from torch.optim import AdamW
from tqdm import tqdm_notebook

from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from configs.pythia_configs_sentiment import get_pythia_config
from reward_class import IMDBSentimentRewardClass, UtilityValuesRewardClass

parser = argparse.ArgumentParser(description='Arguments for RLHF training')
parser.add_argument('--model_name', help='Model name to run PPO on.', default='EleutherAI/pythia-70m')
parser.add_argument('--reward_function', help='Reward function to use.', default='utility_reward')
parser.add_argument('--push_to_hub', action='store_true', help='Push to Hub (True/False)')

class RLHFModelPipeline:
    """
    This class carries out RLHF model training.
    """
    def __init__(self, model_name, reward_function, dataset_name='imdb', push_to_hub=False):
        """
        Initializes model name, reward function, and dataset.
        If you want to push to huggingface hub, set the env variable HUGGINGFACE_ORG_NAME
        """
        self.model_name = model_name
        self.reward_function = reward_function

        self.set_config()
        self.dataset_name = dataset_name
        self.dataset = self.build_dataset()

        huggingface_org_name = os.environ.get('HUGGINGFACE_ORG_NAME', None)

        assert  not ((push_to_hub is True) and huggingface_org_name is None), \
            'If push_to_hub is True, you must specify a Huggingface Org Name under env variable HUGGINGFACE_ORG_NAME'

        self.push_to_hub = push_to_hub
        self.huggingface_org_name = huggingface_org_name


    def set_config(self):
        """
        Sets config for PPO training, including all the relevant hyperparameters.
        """
        self.model_name_simplified = self.model_name.split('/')[-1]
        tracker_project_name = f'trl_{self.model_name_simplified}_{self.reward_function}'

        self.use_adapters = 'gpt-j' in self.model_name

        self.input_min_text_length = 2
        self.input_max_text_length = 10

        # Use smaller batches for large models that need adapters.
        if self.use_adapters:
            batch_size = 32
            mini_batch_size = 8
        else:
            batch_size = 64
            mini_batch_size = 16

        num_warmup_steps = 10
        min_output_length = 8
        max_output_length = 20
        lr = 1e-6

        self.config = get_pythia_config(
            model_name=self.model_name, batch_size=batch_size, mini_batch_size=mini_batch_size, tracker_project_name=tracker_project_name
        )

        if self.reward_function == 'sentiment_reward':
            self.sentiment_reward_class = IMDBSentimentRewardClass()
            print('picked sentiment classifier reward')
        else:
            self.sentiment_reward_class = UtilityValuesRewardClass()
            print('picked utility table reward')

        self.output_length_sampler = LengthSampler(min_output_length, max_output_length)

        if self.use_adapters:
            self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_name, load_in_8bit=False).cuda()
            self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_name, load_in_8bit=False).cuda()
        else:
            self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_name, load_in_8bit=False).cuda()
            self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_name, load_in_8bit=False).cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = AdamW(lr=lr, params=self.policy_model.parameters())

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=self.config.num_training_steps
        )

        self.full_hyperparams_dict = deepcopy(self.config.to_dict())
        self.full_hyperparams_dict.update(
            {
                "min_output_length": min_output_length, "max_output_length": max_output_length,
                "num_training_steps": num_training_steps, "num_warmup_steps": num_warmup_steps
            }
        )


    def build_dataset(self):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.
        """
        # load imdb with datasets
        ds = load_dataset(self.dataset_name, split="train")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_sampler = LengthSampler(self.input_min_text_length, self.input_max_text_length)

        def tokenize(sample):
            sample["input_ids"] = self.tokenizer.encode(sample["review"])[: input_sampler()]
            sample["query"] = self.tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds


    def train(self):
        """
        This function is used to train and (optionally) persist the model to HuggingfaceHub.
        """
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        ppo_trainer = PPOTrainer(
            model=self.policy_model,
            ref_model=self.ref_model,
            config=self.config,
            dataset=self.dataset,
            data_collator=collator,
            lr_scheduler=self.lr_scheduler,
            optimizer=self.optimizer,
            tokenizer=self.tokenizer
        )

        gen_kwargs = {
            "do_sample": True, "min_length": -1, "top_k": 0, "top_p": 1.0,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        self.full_hyperparams_dict.update(gen_kwargs)
        wandb.log(self.full_hyperparams_dict)

        wandb.config.update(self.full_hyperparams_dict)

        for _, input_batch in tqdm_notebook(enumerate(ppo_trainer.dataloader)):
            query_tensors = input_batch["input_ids"]

            #### Get response from gpt2
            response_tensors = []
            print(f'Generating responses for {len(query_tensors)} queries')
            for query_tensor in query_tensors:
                gen_len = self.output_length_sampler()
                gen_kwargs["max_new_tokens"] = gen_len
                response = ppo_trainer.generate(query_tensor, **gen_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])

            print(f'Received {len(response_tensors)} tensors')

            input_batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### Compute sentiment score
            print('Computing sentiment')
            texts = [q + r for q, r in zip(input_batch["query"], input_batch["response"])]
            rewards = self.sentiment_reward_class.assign_rewards(texts)

            #### Run PPO step
            print('Running step')
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, input_batch, rewards)

        #### get a batch from the dataset
        bs = 16
        game_data = {}
        self.dataset.set_format("pandas")
        df_batch = self.dataset[:].sample(bs)
        game_data["query"] = df_batch["query"].tolist()
        query_tensors = df_batch["input_ids"].tolist()


        gen_kwargs['top_k'] = 1
        response_tensors_ref, response_tensors = [], []

        #### get response from gpt2 and gpt2_ref
        for i in range(bs):
            gen_len = 100
            output = self.ref_model.generate(
                torch.tensor(query_tensors[i]).unsqueeze(dim=0).to('cuda'),**gen_kwargs
            ).squeeze()[-gen_len:]
            response_tensors_ref.append(output)
            output = self.policy_model.generate(
                torch.tensor(query_tensors[i]).unsqueeze(dim=0).to('cuda'), **gen_kwargs
            ).squeeze()[-gen_len:]
            response_tensors.append(output)

        #### decode responses
        game_data["response (before)"] = [self.tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
        game_data["response (after)"] = [self.tokenizer.decode(response_tensors[i]) for i in range(bs)]

        #### sentiment analysis of query/response pairs before/after
        texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
        game_data["rewards (before)"] = self.sentiment_reward_class.assign_rewards(texts)

        texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
        game_data["rewards (after)"] = self.sentiment_reward_class.assign_rewards(texts)

        # store results in a dataframe
        df_results = pd.DataFrame(game_data)
        wandb.log(df_results)

        if self.push_to_hub:
            token = os.environ['HUGGINGFACE_HUB_TOKEN']
            login(token=token)
            ppo_trainer.push_to_hub(f"{self.huggingface_org_name}/{self.model_name_simplified}_{self.reward_function}")

        return df_results


if __name__ == '__main__':
    parsed_arguments = parser.parse_args()
    model_name = parsed_arguments.model_name
    reward_function = parsed_arguments.reward_function
    push_to_hub = parsed_arguments.push_to_hub

    rlhf_model_pipeline = RLHFModelPipeline(model_name=model_name, reward_function=reward_function, push_to_hub=True)
    rlhf_model_pipeline
