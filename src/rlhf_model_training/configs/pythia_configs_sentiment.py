"""
Lists valid models and reward functions for Pythia.
"""

from trl import PPOConfig

pythia_model_names = [
    'EleutherAI/pythia-70m', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m'
]

reward_functions = ['sentiment_reward', 'utility_reward']


def get_pythia_config(model_name, batch_size, mini_batch_size, tracker_project_name):
    init_kl_coef = 0.5
    max_grad_norm = 1.0
    # hardcoded for imdb at the moment, and 1 epoch.
    num_training_steps = int(25000 / batch_size)

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
