import wandb

from experiment_configs import ExperimentConfig, experiment_config_A

def run_experiment(experiment_config: ExperimentConfig):
    '''
    Part 1 of IMDb experiment:
    1. Compute parameter divergence and sorts layers by parameter divergence between m_base and m_rlhf.
    2. Extract activations for the train split of IMDb prefixes.
    3. Train autoencoders on the extracted activations.
    4. Measure loss of the autoencoder on the IMDb test dataset.
    '''

    wandb_project_name = 'Autoencoder training'

    wandb.login()
    run = wandb.init(project=wandb_project_name)

    hyperparamters = experiment_config.hyperparameters
    wandb.log(**hyperparameters)

    base_model_name = experiment_config.base_model_name
    policy_model_name = experiment_config.policy_model_name
    
    wandb.log({'base_model_name': base_model_name, 'policy_model_name': policy_model_name})

    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")

    input_data = preprocess(train_dataset, tokenizer, 960).to(device)
    test_data = preprocess(test_dataset, tokenizer, 960).to(device)

    hyperparameters_clone = hyperparameters.copy()
    sorted_layers = find_layers(m_base, m_rlhf)[:3]
    all_autoencoders = {}

    for layer_index in sorted_layers:
        layer_name = f"layers.{layer_index}.mlp"
        print(f"Layer: {layer_name}")

        for hidden_size in hidden_sizes:
            print(f"Training autoencoder with hidden size: {hidden_size}")
            hyperparameters_clone['hidden_size'] = hidden_size

            autoencoders = feature_representation(m_base, layer_name, input_data, hyperparameters_clone, device)

            if layer_name not in all_autoencoders:
                all_autoencoders[layer_name] = []

            all_autoencoders[layer_name].extend(autoencoders)

            test_activations = get_layer_activations(m_base, layer_name, test_data, device)
            test_activations_tensor = test_activations.detach().clone().to(device)
            test_activations_tensor = test_activations_tensor.squeeze(1)
            test_dataset = TensorDataset(test_activations_tensor)
            test_data_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

            for autoencoder in autoencoders:
                test_loss = measure_performance(autoencoder, test_data_loader, device)
                print(f'Test Loss for Autoencoder with hidden size {hidden_size}: {test_loss:.4f}')


run_experiment
