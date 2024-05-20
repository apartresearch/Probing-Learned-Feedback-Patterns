"""
This class runs the actual experiments
"""
import random

import wandb

from datasets import load_dataset
from transformers import AutoModel
from transformers import AutoTokenizer

from reward_analyzer.sparse_codes_training.experiment_configs import ExperimentConfig
from reward_analyzer.configs.task_configs import TaskConfig
from reward_analyzer.sparse_codes_training.experiment_helpers.autoencoder_trainer_and_preparer import AutoencoderDataPreparerAndTrainer
from reward_analyzer.sparse_codes_training.experiment_helpers.layer_activations_handler import LayerActivationsHandler
from reward_analyzer.sparse_codes_training.metrics.mmcs import compare_autoencoders

from reward_analyzer.utils.gpu_utils import  find_gpu_with_most_memory
from reward_analyzer.utils.model_storage_utils import save_autoencoders_for_artifact, load_latest_model_from_hub

class ExperimentRunner:
    """
    Given an experiment trl_config, instantiates the models and datasets,
    and instantiates calls to find divergent layers between policy and rlhf modela,
    and train autoencoders for each such layer.
    """

    def __init__(self, experiment_config: ExperimentConfig, seed: int = 42):
        self.experiment_config = experiment_config
        self.initialize_run_and_hyperparameters(experiment_config)
        self.task_config = experiment_config.task_config
        self.seed = seed

        self.m_base, self.m_rlhf, self.tokenizer, self.autoencoder_device = self.initialize_models(
            base_model_name=self.base_model_name, task_config=experiment_config.task_config,
            model_device=self.model_device
        )

        self.activations_handler = LayerActivationsHandler(model=self.m_base)

        self.initialize_datasets()
        self.initialize_autoencoder_trainers_and_holders()

        self.sorted_layers, self.divergences_by_layer = [], []

    def initialize_autoencoder_trainers_and_holders(self):
        """
        Initializes feature extractors, as well as the holder classes
        that contain the final collection of trained autoencoders.
        """
        # Autoencoders of larger hidden size for base model
        self.autoencoders_base_big = {}

        # Autoencoders of smaller hidden size for base model
        self.autoencoders_base_small = {}

        # Autoencoders of larger hidden size for base model
        self.autoencoders_rlhf_big = {}

        # Autoencoders of smaller hidden size for policy model
        self.autoencoders_rlhf_small = {}

        self.ae_extractor_base = AutoencoderDataPreparerAndTrainer(
            model=self.m_base, tokenizer=self.tokenizer, hyperparameters=self.hyperparameters,
            autoencoder_device=self.autoencoder_device
        )

        self.ae_extractor_rlhf = AutoencoderDataPreparerAndTrainer(
            model=self.m_rlhf, tokenizer=self.tokenizer, hyperparameters=self.hyperparameters,
            autoencoder_device=self.autoencoder_device
        )

    def initialize_run_and_hyperparameters(self, experiment_config: ExperimentConfig):
        """
        This initializes the wandb run, the hyperparameters, model names and datasets.
        """
        wandb.login()

        self.hyperparameters = self.experiment_config.hyperparameters
        self.base_model_name = experiment_config.base_model_name
        self.policy_model_name = experiment_config.policy_model_name

        self.wandb_project_name = experiment_config.wandb_project_name

        self.hyperparameters.update(
            {'base_model_name': self.base_model_name, 'policy_model_name': self.policy_model_name}
        )

        self.run = wandb.init(project=self.wandb_project_name, config=self.hyperparameters)

        self.is_fast = self.hyperparameters['fast']
        self.input_device = self.experiment_config.device
        self.model_device = self.input_device if self.input_device else find_gpu_with_most_memory()

        self.num_layers_to_keep = self.hyperparameters['num_layers_to_keep']
        self.hidden_size_multiples = sorted(self.hyperparameters['hidden_size_multiples'].copy())
        self.small_hidden_size_multiple = self.hidden_size_multiples[0]


        if 'pythia' in self.base_model_name:
            self.layer_name_stem = 'layers'
        elif ('gpt-neo' in self.base_model_name) or ('gpt-j' in self.base_model_name):
            self.layer_name_stem = 'h'
        elif ('gemma' in self.base_model_name):
            self.layer_name_stem = 'mlp'
        else:
            raise Exception(f'Unsupported model type {self.base_model_name}')

    def initialize_models(
        self, base_model_name: str, task_config: TaskConfig, model_device: str = find_gpu_with_most_memory()
    ):
        """
        Initialize base and policy models.
        """
        task_name = task_config.name
        m_base = AutoModel.from_pretrained(base_model_name, load_in_8bit=True).to(model_device)
        m_rlhf = load_latest_model_from_hub(model_name=base_model_name, task_config=task_config).to(model_device)

        # We may need to train autoencoders on different device after loading models.
        autoencoder_device = self.input_device if self.input_device else find_gpu_with_most_memory()

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        return m_base, m_rlhf, tokenizer, autoencoder_device

    def initialize_datasets(self):
        """
        Initializes the datasets using hyperparameters.
        """
        self.split = self.hyperparameters['split']
        print('Processing texts')


        if self.task_config == TaskConfig.IMDB:
            print(f'Loading imdb dataset for {self.task_config.name}')
            self.dataset_name = 'imdb'
            self.test_dataset_base = load_dataset(self.dataset_name, split=self.split)
            self.test_dataset_base = [x['text'] for x in self.test_dataset_base]

        elif self.task_config in [TaskConfig.HH_RLHF, TaskConfig.UNALIGNED]:
            print(f'Loading anthropic dataset for {self.task_config.name}')
            self.dataset_name = 'anthropic/hh-rlhf'

            self.test_dataset_base = load_dataset(self.dataset_name, split=self.split)
            result_dataset = []
            for item in self.test_dataset_base:
                result_dataset.extend([item['chosen'], item['rejected']])

            random.seed(self.seed)

            result_dataset = random.sample(result_dataset, k=75000)
            self.test_dataset_base = result_dataset

        else:
            raise Exception(f'Parsing dataset {self.dataset_name} is not supported')

        if self.is_fast:
            self.hyperparameters['batch_size'] = 4
            self.test_dataset_base = self.test_dataset_base[:12]

        self.test_dataset_rlhf = self.test_dataset_base.copy()
        self.num_examples = len(self.test_dataset_base)
        print(f'Working with {self.num_examples} texts.')

        wandb.run.config['seed'] = self.seed
        wandb.run.config['num_examples'] = self.num_examples
        self.hyperparameters['num_examples'] = self.num_examples


    def find_sorted_divergent_layers(self, num_layers_to_keep: int):
        """
        Finds most divergence layers between base_model and rlhf_model
        """
        sorted_layers, divergences_by_layer = self.activations_handler.find_divergences(
            other_model=self.m_rlhf, hyperparameters=self.hyperparameters)
        wandb.config['sorted_layers'] = sorted_layers

        sorted_layers = sorted_layers[:num_layers_to_keep]
        return sorted_layers, divergences_by_layer

    def extract_autoencoder_for_base_and_rlhf_at_layer_index(
            self, hidden_size_multiple: int, layer_index: int, label: str
    ):
        """
        Extracts autoencoders for a given layer index from base and rlhf model.
        """

        print(f'Training base model autoencoder')
        autoencoder_base = self.ae_extractor_base.train_autoencoder_on_text_activations(
            layer_name=f'{self.layer_name_stem}.{layer_index}.mlp',
            input_texts=self.test_dataset_base, hidden_size_multiple=hidden_size_multiple,
            label=f'base_{label}'
        )

        print(f'Training rlhf model autoencoder')
        autoencoder_rlhf = self.ae_extractor_rlhf.train_autoencoder_on_text_activations(
            layer_name=f'{self.layer_name_stem}.{layer_index}.mlp',
            input_texts=self.test_dataset_rlhf, hidden_size_multiple=hidden_size_multiple,
            label=f'rlhf_{label}'
        )

        target_autoencoders_base = (
            self.autoencoders_base_big if hidden_size_multiple > self.small_hidden_size_multiple
            else self.autoencoders_base_small
        )
        target_autoencoders_base[str(layer_index)] = autoencoder_base

        target_autoencoders_rlhf = (
            self.autoencoders_rlhf_big if hidden_size_multiple > self.small_hidden_size_multiple
            else self.autoencoders_rlhf_small
        )
        target_autoencoders_rlhf[str(layer_index)] = autoencoder_rlhf

    def run_experiment(self):
        """
        With the hyperparameters, models and datasets already set.
        This function runs the end to end training, finding the
        divergent layers, training autoencoder pairs at each,
        computing mmcs, and finally saving the created artifacts to wandb.
        """
        self.sorted_layers, self.divergences_by_layer = self.find_sorted_divergent_layers(
            num_layers_to_keep=self.num_layers_to_keep
        )

        # Create autoencoder pairs for each layer
        for number, layer_index in enumerate(self.sorted_layers):
            print(f'Training autoencoder pair # {number}')

            # For each pair, set the hidden size to be a different multiple of the input size
            for hidden_size_multiple in self.hidden_size_multiples:
                label = 'big' if hidden_size_multiple > self.small_hidden_size_multiple else 'small'

                self.extract_autoencoder_for_base_and_rlhf_at_layer_index(
                    hidden_size_multiple=hidden_size_multiple, layer_index=layer_index, label=label
                )

        # Compare overlaps between large and small autoencoder feature dictionaries.
        base_mmcs_results = compare_autoencoders(
            small_dict=self.autoencoders_base_small, big_dict=self.autoencoders_base_big
        )
        rlhf_mmcs_results = compare_autoencoders(
            small_dict=self.autoencoders_rlhf_small, big_dict=self.autoencoders_rlhf_big
        )

        added_metadata = {}

        mmcs_results = {"base_mmcs_results": base_mmcs_results, "rlhf_mmcs_results": rlhf_mmcs_results}
        divergences_by_layer = {'divergences_by_layer': self.divergences_by_layer}

        added_metadata.update(mmcs_results)
        added_metadata.update(divergences_by_layer)
        wandb.log(added_metadata)

        print('saving to wandb')
        # Finally, log to wandb.
        save_autoencoders_for_artifact(
            autoencoders_base_big=self.autoencoders_base_big, autoencoders_base_small=self.autoencoders_base_small,
            autoencoders_rlhf_big=self.autoencoders_rlhf_big, autoencoders_rlhf_small=self.autoencoders_rlhf_small,
            policy_model_name=self.policy_model_name, hyperparameters=self.hyperparameters,
            alias='latest', run=self.run, added_metadata=added_metadata
        )
        wandb.finish()
