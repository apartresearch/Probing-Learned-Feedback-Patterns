import wandb

from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

from experiment_configs import ExperimentConfig, all_experiment_configs

from network_helper_functions import find_layers
from training import feature_representation
from utils.model_storage_utils import save_autoencoders_for_artifact

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess(dataset, tokenizer, limit=None):
    if limit:
        texts = [x['text'] for x in dataset.select(range(limit))]
    else:
        texts = [x['text'] for x in dataset]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs

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

    simplified_policy_model_name = policy_model_name.split('/')[-1].replace('-', '_')
    wandb_project_name = f'Autoencoder_training_{simplified_policy_model_name}'

    hyperparameters.update({'base_model_name': base_model_name, 'policy_model_name': policy_model_name})
    run = wandb.init(project=wandb_project_name, config=hyperparameters)

    base_model_name = experiment_config.base_model_name
    policy_model_name = experiment_config.policy_model_name

    m_base = AutoModel.from_pretrained(base_model_name).to(device)
    m_rlhf = AutoModel.from_pretrained(policy_model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer_rlhf = AutoTokenizer.from_pretrained(policy_model_name)
    tokenizer_rlhf.pad_token = tokenizer_rlhf.eos_token

    split = hyperparameters['split']

    if is_fast:
        test_dataset_base = preprocess(load_dataset("imdb", split=split), tokenizer, 96)
        test_dataset_rlhf = preprocess(load_dataset("imdb", split=split), tokenizer_rlhf, 96)
    else:
        test_dataset_base = preprocess(load_dataset("imdb", split=split), tokenizer)
        test_dataset_rlhf = preprocess(load_dataset("imdb", split=split), tokenizer_rlhf)

    input_data_base = {'input_ids': test_dataset_base['input_ids'].to(device)}
    input_data_rlhf = {'input_ids': test_dataset_rlhf['input_ids'].to(device)}

    num_examples = len(test_dataset_base)
    wandb.run.config['num_examples'] = num_examples
    hyperparameters['num_examples'] = num_examples
    sorted_layers = find_layers(m_base, m_rlhf)

    autoencoders_base_big = {}
    autoencoders_base_small = {}

    autoencoders_rlhf_big = {}
    autoencoders_rlhf_small = {}

    hidden_size_multiples = sorted(hyperparameters['hidden_size_multiples'].copy())
    small_hidden_size_multiple = hidden_size_multiples[0]

    for layer_index in sorted_layers:
        for hidden_size_multiple in hidden_size_multiples:
            hyperparameters_copy = hyperparameters.copy()
            hyperparameters_copy['hidden_size_multiple'] = hidden_size_multiple

            label = 'big' if hidden_size_multiple > small_hidden_size_multiple else 'small'

            autoencoder_base = feature_representation(
                m_base, f'layers.{sorted_layers[layer_index]}.mlp',
                input_data_base, hyperparameters_copy, device, label=f'base_{label}'
            )

            target_autoencoders_base = autoencoders_base_big if hidden_size_multiple > small_hidden_size_multiple else autoencoders_base_small
            target_autoencoders_base[str(layer_index)] = autoencoder_base

            autoencoder_rlhf = feature_representation(
                m_rlhf, f'layers.{sorted_layers[layer_index]}.mlp',
                input_data_rlhf, hyperparameters_copy, device, label=f'rlhf_{label}'
            )

            target_autoencoders_rlhf = autoencoders_rlhf_big if hidden_size_multiple > small_hidden_size_multiple else autoencoders_rlhf_small
            target_autoencoders_rlhf[str(layer_index)] = autoencoder_rlhf

    save_autoencoders_for_artifact(
        autoencoders_base_big=autoencoders_base_big, autoencoders_base_small=autoencoders_base_small,
        autoencoders_rlhf_big=autoencoders_rlhf_big, autoencoders_rlhf_small=autoencoders_rlhf_small,
        policy_model_name=policy_model_name, hyperparameters=hyperparameters, alias='latest', run=run
    )
    wandb.finish()

for experiment_config in all_experiment_configs:
    print(f'Running experiment now for config {experiment_config}')
    run_experiment(experiment_config=experiment_config)
