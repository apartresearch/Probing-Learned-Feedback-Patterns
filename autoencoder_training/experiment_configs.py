# Will increment hyperparameter sets as we try different ones.

class ExperimentConfig:

    def __init__(self, hyperparameters, base_model_name, policy_model_name, device=None):
        self.hyperparameters = hyperparameters
        self.base_model_name = base_model_name
        self.policy_model_name = policy_model_name
        self.device = device

    def __str__(self):
        printable = self.hyperparameters.copy()
        printable.update({'base_model_name': self.base_model_name, 'policy_model_name': self.policy_model_name})
        return str(printable)

hyperparameters_1 = {
    'hidden_size_multiples': [1, 2],
    'l1_coef': 0.001,
    'batch_size': 32,
    'num_epochs': 3,
    'learning_rate': 1e-3,
    'fast': True,
    'split': 'test',
    'num_layers_to_keep': 5
}


hyperparameters_2 = {
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

all_models = ['eleutherai/pythia-70m', 'eleutherai/pythia-160m', 'eleutherai/pythia-410m', 'eleutherai/gpt-neo-125m']
all_reward_functions = ['sentiment_reward', 'utility_reward']

def generate_experiment_configs(hyperparameters):
    grid_experiment_configs = {}
    for model_name in all_models:
        for reward_function in all_reward_functions:
            simplified_model_name = model_name.split('/')[-1]
            policy_model_name = f'amirabdullah19852020/{simplified_model_name}_{reward_function}'
            new_config = ExperimentConfig(hyperparameters=hyperparameters, base_model_name=model_name, policy_model_name=policy_model_name)

            experiment_key = (simplified_model_name, reward_function)
            grid_experiment_configs[experiment_key] = new_config
    return grid_experiment_configs

grid_experiment_configs = generate_experiment_configs(hyperparameters_2)
