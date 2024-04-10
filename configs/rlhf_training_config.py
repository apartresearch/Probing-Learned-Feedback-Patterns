"""
Lists valid models and reward functions for Pythia.
"""
from datasets import Dataset
from trl import PPOConfig

class RLHFTrainingConfig:
    """
    Specifies configs used for each stage of training RLHF models.
    """

    def get_model_config(self, model_name: str, dataset: Dataset, batch_size: int, mini_batch_size: int, tracker_project_name: str):
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
                model_name=model_name,
                tracker_project_name=tracker_project_name,
                steps=num_training_steps,
            )
            return config

        elif 'neo' in model_name:
            pass
        elif 'llama' in model_name:
            pass

        elif 'mistral' in model_name:
            pass
