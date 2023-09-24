# Will increment hyperparameter sets as we try different ones.

class ExperimentConfig:

    def __init__(self, hyperparameters, base_model_name, policy_model_name):
        self.hyperparameters = hyperparameters
        self.base_model_name = base_model_name
        self.policy_model_name = policy_model_name

hyperparameters_1 = {
    'input_size': 512,
    'hidden_sizes': [512, 1024],
    'l1_coef': 0.1,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3
}


hyperparameters_2 = {
    'input_size': 512,
    'hidden_sizes': [512, 1024],
    'l1_coef': 0.01,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3
}


experiment_config_A = ExperimentConfig(hyperparameters=hyperparameters_1,  base_model_name="eleutherai/pythia-70m", policy_model_name="amirabdullah19852020/pythia-70m_sentiment_reward")
