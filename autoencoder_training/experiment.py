import argparse
import wandb
from datasets import load_dataset

from transformers import AutoModel
from transformers import AutoTokenizer


from experiment_configs import (
    ExperimentConfig, grid_experiment_configs
)

from network_helper_functions import find_layers
from training import feature_representation

from utils.gpu_utils import find_gpu_with_most_memory
from utils.model_storage_utils import save_autoencoders_for_artifact

parser = argparse.ArgumentParser(description="Choose which experiment config you want to run.")
parser.add_argument("--base_model_name", default='pythia-70m', type=str, help="The model name you want to use.", required=False)
parser.add_argument("--reward_function", default='utility_reward', type=str, help="The reward function you want to leverage.", required=False)

def run_experiment(experiment_config: ExperimentConfig):
    '''
    Part 1 of IMDb experiment:
    1. Compute parameter divergence and sorts layers by parameter divergence between m_base and m_rlhf.
    2. Extract activations for the train split of IMDb prefixes.
    3. Train autoencoders on the extracted activations.
    4. Measure loss of the autoencoder on the IMDb test dataset.
    '''
    wandb.login()
    hyperparameters = experiment_config.hyperparameters
    base_model_name = experiment_config.base_model_name
    policy_model_name = experiment_config.policy_model_name

    is_fast = hyperparameters['fast']
    input_device = experiment_config.device
    num_layers_to_keep = hyperparameters['num_layers_to_keep']

    device = input_device if input_device else find_gpu_with_most_memory()

    simplified_policy_model_name = policy_model_name.split('/')[-1].replace('-', '_')
    wandb_project_name = f'Autoencoder_training_{simplified_policy_model_name}'

    hyperparameters.update({'base_model_name': base_model_name, 'policy_model_name': policy_model_name})
    run = wandb.init(project=wandb_project_name, config=hyperparameters)

    base_model_name = experiment_config.base_model_name
    policy_model_name = experiment_config.policy_model_name

    if 'pythia' in base_model_name:
        layer_name_stem = 'layers'
    elif ('gpt-neo' in base_model_name) or ('gpt-j' in base_model_name):
        layer_name_stem = 'h'
    else:
        raise Exception(f'Unsupported model type {base_model_name}')

    if 'gpt-j' in policy_model_name:
        m_base = AutoModel.from_pretrained(base_model_name, device_map="auto")
        m_rlhf = AutoModel.from_pretrained(base_model_name, device_map="auto")
        m_rlhf.load_adapter(policy_model_name)

    else:
        m_base = AutoModel.from_pretrained(base_model_name).to(device)
        m_rlhf = AutoModel.from_pretrained(policy_model_name).to(device)

    # We may need to train autoencoders on different device after loading models.
    device = input_device if input_device else find_gpu_with_most_memory()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer_rlhf = AutoTokenizer.from_pretrained(policy_model_name)
    tokenizer_rlhf.pad_token = tokenizer_rlhf.eos_token

    split = hyperparameters['split']
    print('Processing texts')

    if is_fast:
        hyperparameters['batch_size'] = 4
        test_dataset_base = load_dataset("imdb", split=split).select(range(12))
        test_dataset_base = [x['text'] for x in test_dataset_base]
        test_dataset_rlhf = test_dataset_base.copy()
    else:
        test_dataset_base = load_dataset("imdb", split=split)
        test_dataset_base = [x['text'] for x in test_dataset_base]
        test_dataset_rlhf = test_dataset_base.copy()

    num_examples = len(test_dataset_base)

    print(f'Working with {num_examples} texts.')

    wandb.run.config['num_examples'] = num_examples
    hyperparameters['num_examples'] = num_examples
    sorted_layers, divergences_by_layer = find_layers(m_base, m_rlhf)
    wandb.config['sorted_layers'] = sorted_layers

    sorted_layers = sorted_layers[:num_layers_to_keep]

    autoencoders_base_big = {}
    autoencoders_base_small = {}

    autoencoders_rlhf_big = {}
    autoencoders_rlhf_small = {}

    hidden_size_multiples = sorted(hyperparameters['hidden_size_multiples'].copy())
    small_hidden_size_multiple = hidden_size_multiples[0]

    for position, layer_index in enumerate(sorted_layers):
        for hidden_size_multiple in hidden_size_multiples:
            hyperparameters_copy = hyperparameters.copy()
            hyperparameters_copy['hidden_size_multiple'] = hidden_size_multiple

            label = 'big' if hidden_size_multiple > small_hidden_size_multiple else 'small'

            autoencoder_base = feature_representation(
                model=m_base, tokenizer=tokenizer, layer_name=f'{layer_name_stem}.{layer_index}.mlp',
                input_texts= test_dataset_base, hyperparameters=hyperparameters_copy, device=device, label=f'base_{label}'
            )

            target_autoencoders_base = autoencoders_base_big if hidden_size_multiple > small_hidden_size_multiple else autoencoders_base_small
            target_autoencoders_base[str(layer_index)] = autoencoder_base

            print(f'Working with {layer_index} of position {position} in {label}.')

            autoencoder_rlhf = feature_representation(
                model=m_rlhf, tokenizer=tokenizer, layer_name=f'{layer_name_stem}.{layer_index}.mlp',
                input_texts=test_dataset_rlhf, hyperparameters=hyperparameters_copy, device=device, label=f'rlhf_{label}'
            )

            target_autoencoders_rlhf = autoencoders_rlhf_big if hidden_size_multiple > small_hidden_size_multiple else autoencoders_rlhf_small
            target_autoencoders_rlhf[str(layer_index)] = autoencoder_rlhf

    save_autoencoders_for_artifact(
        autoencoders_base_big=autoencoders_base_big, autoencoders_base_small=autoencoders_base_small,
        autoencoders_rlhf_big=autoencoders_rlhf_big, autoencoders_rlhf_small=autoencoders_rlhf_small,
        policy_model_name=policy_model_name, hyperparameters=hyperparameters, alias='latest', run=run,
        added_metadata={'divergences_by_layer': divergences_by_layer}
    )
    wandb.finish()


args = parser.parse_args()
base_model_name = args.base_model_name
reward_function = args.reward_function
chosen_experiment_config = grid_experiment_configs[(base_model_name, reward_function)]

print(f'Running experiment now for config {chosen_experiment_config}')
run_experiment(experiment_config=chosen_experiment_config)
