"""
Lists valid models and reward functions for Pythia.
"""
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from trl import PPOConfig

class PPOTrainingConfig:
    """
    Specifies configs used for each stage of training RLHF models.
    """

    def get_model_config(self, model_name: str, dataset: Dataset, tracker_project_name: str):
        # Use smaller batches for large models that need adapters.
        batch_size = 64
        mini_batch_size = 16
        num_warmup_steps = 10
        lr = 1e-6
        if 'pythia' in model_name:
            init_kl_coef = 0.5
            max_grad_norm = 1.0
            # hardcoded for imdb at the moment, and 1 epoch.
            num_training_steps = int(len(dataset) / batch_size)

            config = PPOConfig(
                batch_size=batch_size,
                init_kl_coef=init_kl_coef,
                log_with="wandb",
                max_grad_norm=max_grad_norm,
                mini_batch_size=mini_batch_size,
                learning_rate=lr,
                model_name=model_name,
                tracker_project_name=tracker_project_name,
                steps=num_training_steps,
            )
            config.num_warmup_steps = num_warmup_steps
            return config

        elif 'neo' in model_name:
            pass
        elif 'llama' in model_name:
            pass

        elif 'mistral' in model_name:
            pass


@dataclass
class DPOTrainingConfig:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="EleutherAI/gpt-neo-125m", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1.5e-4, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=256, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=20050, metadata={"help": "max number of training steps"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )

    huggingface_hub_name: Optional[str] = field(
        default=None, metadata={"help": "Huggingface repo name"}
    )
