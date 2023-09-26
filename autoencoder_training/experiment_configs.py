# Will increment hyperparameter sets as we try different ones.

class ExperimentConfig:

    def __init__(self, hyperparameters, base_model_name, policy_model_name):
        self.hyperparameters = hyperparameters
        self.base_model_name = base_model_name
        self.policy_model_name = policy_model_name

    def __str__(self):
        printable = self.hyperparameters.copy()
        printable.update({'base_model_name': self.base_model_name, 'policy_model_name': self.policy_model_name})
        return str(printable)

hyperparameters_1 = {
    'hidden_size_multiples': [1, 2],
    'l1_coef': 0.1,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'fast': False,
    'split': 'test',
    'device': '3'
}


hyperparameters_2 = {
    'hidden_size_multiples': [1, 2],
    'l1_coef': 0.01,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'fast': False,
    'split': 'test',
    'device': '3'
}

all_models = ['eleutherai/pythia-70m', 'eleutherai/pythia-160m', 'eleutherai/pythia-410m']
all_reward_functions = ['sentiment_reward', 'utility_reward']

def generate_experiment_configs(hyperparameters):
    all_experiment_configs = []
    for model_name in all_models:
        for reward_function in all_reward_functions:
            simplified_model_name = model_name.split('/')[-1]
            policy_model_name = f'amirabdullah19852020/{simplified_model_name}_{reward_function}'
            new_config = ExperimentConfig(hyperparameters=hyperparameters, base_model_name=model_name, policy_model_name=policy_model_name)
            all_experiment_configs.append(new_config)
    return all_experiment_configs


all_experiment_configs = generate_experiment_configs(hyperparameters_2)

experiment_config_A = ExperimentConfig(hyperparameters=hyperparameters_1,  base_model_name="eleutherai/pythia-70m", policy_model_name="amirabdullah19852020/pythia-70m_sentiment_reward")
