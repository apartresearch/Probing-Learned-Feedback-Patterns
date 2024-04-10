"""
This class carries out the RLHF process on a base model.
In particular, over here we do RLHF for completing prefixes of IMDB dataset with positive sentiment.
"""
import os
from copy import deepcopy
from abc import abstractmethod

import huggingface_hub

from datasets import Dataset
from torch.optim import AdamW

from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from trl import AutoModelForCausalLMWithValueHead

from configs.rlhf_training_config import RLHFTrainingConfig
from utils.gpu_utils import find_gpu_with_most_memory

class RLHFModelPipeline:
    """
    This class carries out RLHF model training.
    """
    def __init__(self, model_name, dataset_name='imdb', push_to_hub=False):
        """
        Initializes model name, reward function, and dataset.
        If you want to push to huggingface hub, set the env variable HUGGINGFACE_ORG_NAME
        """
        self.model_name = model_name

        self.device = find_gpu_with_most_memory()
        self.full_hyperparams_dict = {}
        self.rlhf_training_config = RLHFTrainingConfig()

        self.model_name_simplified = self.model_name.split('/')[-1]
        self.tracker_project_name = f'trl_{self.model_name_simplified}_rlhf_training'
        self.trl_trainer = None

        self.set_config_and_model()
        self.dataset_name = dataset_name
        self.dataset = self.build_dataset_and_reward()

        huggingface_org_name = os.environ.get('HUGGINGFACE_ORG_NAME', None)
        assert not ((push_to_hub is True) and huggingface_org_name is None), \
            'If push_to_hub is True, you must specify a Huggingface Org Name under env variable HUGGINGFACE_ORG_NAME'

        self.push_to_hub = push_to_hub
        self.huggingface_org_name = huggingface_org_name

    def set_config_and_model(self):
        """
        Sets trl_config for PPO training, including all the relevant hyperparameters.
        """

        self.use_adapters = 'gpt-j' in self.model_name

        # Use smaller batches for large models that need adapters.
        if self.use_adapters:
            batch_size = 32
            mini_batch_size = 8
        else:
            batch_size = 64
            mini_batch_size = 16

        num_warmup_steps = 10
        lr = 1e-6

        self.trl_config = self.rlhf_training_config.get_model_config(
            model_name=self.model_name, batch_size=batch_size, mini_batch_size=mini_batch_size,
            tracker_project_name=self.tracker_project_name
        )

        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name, load_in_8bit=False).cuda(device=self.device)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name, load_in_8bit=False).cuda(device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = AdamW(lr=lr, params=self.policy_model.parameters())

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=self.trl_config.steps
        )

        self.full_hyperparams_dict = deepcopy(self.trl_config.to_dict())
        self.full_hyperparams_dict.update(
            {
                "num_training_steps": self.trl_config.steps, "num_warmup_steps": num_warmup_steps
            }
        )

    @abstractmethod
    def build_dataset_and_reward(self) -> Dataset:
        """
        Builds the dataset and sets the reward function for pipeline.
        """

    def push_results_to_hub(self):
        if self.push_to_hub:
            token = os.environ['HUGGINGFACE_HUB_TOKEN']
            huggingface_hub.login(token=token)
            self.trl_trainer.push_to_hub(f"{self.huggingface_org_name}/{self.model_name_simplified}_{self.dataset_name}_reward")