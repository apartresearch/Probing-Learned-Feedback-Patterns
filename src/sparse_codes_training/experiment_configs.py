"""
This module gives the experiment configs for a given run.
We define
"""

class ExperimentConfig:
    """
    This fully specifies one run of our experiment extracting base and policy models.
    """

    def __init__(
            self, hyperparameters, base_model_name, policy_model_name,
            wandb_project_name=None, device=None
    ):
        self.hyperparameters = hyperparameters
        self.base_model_name = base_model_name
        self.policy_model_name = policy_model_name

        if wandb_project_name:
            self.wandb_project_name = wandb_project_name
        else:
            simplified_policy_model_name = self.policy_model_name.split('/')[-1].replace('-', '_')
            self.wandb_project_name = f'Autoencoder_training_{simplified_policy_model_name}'
        self.device = device

    def __str__(self):
        printable = self.hyperparameters.copy()
        printable.update({'base_model_name': self.base_model_name, 'policy_model_name': self.policy_model_name})
        return str(printable)

hyperparameters_fast = {
    'hidden_size_multiples': [1, 2],
    'l1_coef': 0.001,
    'batch_size': 32,
    'num_epochs': 1,
    'learning_rate': 1e-3,
    'fast': True,
    'split': 'test',
    'num_layers_to_keep': 5,
    'tied_weights': True
}


hyperparameters_full = {
    'max_input_length': 256,
    'hidden_size_multiples': [1, 2],
    'l1_coef': 0.001,
    'batch_size': 32,
    'num_epochs': 3,
    'learning_rate': 1e-3,
    'fast': False,
    'split': 'test',
    'num_layers_to_keep': 5,
    'tied_weights': True
}

all_models = [
    'eleutherai/pythia-70m', 'eleutherai/pythia-160m', 'eleutherai/pythia-410m',
    'eleutherai/gpt-neo-125m', 'ybelkada/gpt-j-6b-sharded-bf16'
]

all_reward_functions = ['sentiment_reward', 'utility_reward']

model_specific_parameters = {
  'pythia-70m': {},
  'pythia-160m': {},
  'pythia-410m': {},
  'gpt-neo-125m': {'l1_coef': 0.015},
  'gpt-j-6b-sharded-bf16': {'batch_size': 8, 'num_epochs': 1, 'gradient_accumulation_steps': 4}
}

def generate_experiment_configs(hyperparameters):
    """
    We generate experiment configs as a cross product of all possible
    model names and tasks, with the given hyperparameters.
    We update some model specific values based on experimentation -
    namely, gpt-neo-125 has a different l1_coef of 0.015 based on our
    experiments.
    """
    all_experiment_configs = {}
    for model_name in all_models:
        for reward_function in all_reward_functions:
            simplified_model_name = model_name.rsplit('/', maxsplit=1)[-1]
            policy_model_name = f'amirabdullah19852020/{simplified_model_name}_{reward_function}'
            hyperparameters_copy = hyperparameters.copy()
            hyperparameters_copy.update(model_specific_parameters[simplified_model_name])

            new_config = ExperimentConfig(
                hyperparameters=hyperparameters_copy, base_model_name=model_name,
                policy_model_name=policy_model_name
            )

            experiment_key = (simplified_model_name, reward_function)
            all_experiment_configs[experiment_key] = new_config
    return all_experiment_configs

grid_experiment_configs = generate_experiment_configs(hyperparameters_full)
