import pandas as pd
import torch
import wandb
from tqdm import tqdm
from trl import PPOTrainer
from trl.core import LengthSampler

from rlhf_model_training.model_pipelines.rlhf_model_pipeline import RLHFModelPipeline
from rlhf_model_training.reward_class import UtilityValuesRewardClass

class IMDBTrainingPipeline(RLHFModelPipeline):
    """
    Extends RLHFModelPipeline
    """

    def train(self):
        """
        This function is used to train and (optionally) persist the model to HuggingfaceHub.
        """
        min_output_length = 8
        max_output_length = 20
        self.output_length_sampler = LengthSampler(min_output_length, max_output_length)
        self.reward_class = UtilityValuesRewardClass()
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        self.trl_trainer = PPOTrainer(
            model=self.policy_model,
            ref_model=self.ref_model,
            config=self.trl_config,
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

        for _, input_batch in tqdm(enumerate(self.trl_trainer.dataloader), total=self.trl_config.steps):
            query_tensors = input_batch["input_ids"]

            #### Get response from gpt2
            response_tensors = []
            print(f'Generating responses for {len(query_tensors)} queries')
            for query_tensor in query_tensors:
                gen_len = self.output_length_sampler()
                gen_kwargs["max_new_tokens"] = gen_len
                response = self.trl_trainer.generate(query_tensor, **gen_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])

            print(f'Received {len(response_tensors)} tensors')

            input_batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### Compute sentiment score
            print('Computing sentiment')
            texts = [r for r in input_batch["response"]]
            rewards = self.reward_class.assign_rewards(texts)

            #### Run PPO step
            print('Running step')
            stats = self.trl_trainer.step(query_tensors, response_tensors, rewards)
            self.trl_trainer.log_stats(stats, input_batch, rewards)

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
        game_data["rewards (before)"] = self.reward_class.assign_rewards(texts)

        texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
        game_data["rewards (after)"] = self.reward_class.assign_rewards(texts)

        self.push_to_hub()

        # store results in a dataframe
        df_results = pd.DataFrame(game_data)
        wandb.log(df_results)

        return df_results