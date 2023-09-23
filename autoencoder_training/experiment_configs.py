# Will increment hyperparameter sets as we try different ones.
hyperparameters_1 = {
    'input_size': 512,
    'hidden_sizes': [512, 1024],
    'sparsity_target': 0.1,
    'sparsity_weight': 1e-2,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3
}


hyperparameters_2 = {
    'input_size': 512,
    'hidden_sizes': [512, 1024],
    'sparsity_target': 0.1,
    'sparsity_weight': 1e-1,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3
}
