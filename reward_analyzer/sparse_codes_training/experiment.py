"""
Entrypoint for code from which experiments are triggered/launched.
"""
import argparse
from time import sleep

from reward_analyzer.sparse_codes_training.experiment_configs import (
    ExperimentConfig, grid_experiment_configs
)
from reward_analyzer.sparse_codes_training.experiment_helpers.experiment_runner import ExperimentRunner
from reward_analyzer.configs.task_configs import TaskConfig

parser = argparse.ArgumentParser(description="Choose which experiment trl_config you want to run.")

config_names_to_tasks = {
    "imdb": TaskConfig.IMDB,
    "unaligned": TaskConfig.UNALIGNED,
    "hh": TaskConfig.HH_RLHF
}

parser.add_argument(
    "--fast", action="store_true", help="Whether to run in fast mode or not.", required=False)
parser.add_argument(
    "--dataset", default="imdb", type=str, help="The dataset to train the autoencoder on", required=False
)
parser.add_argument(
    "--split", default="test", type=str, help="The dataset split to train the autoencoder on", required=False
)
parser.add_argument(
    "--tied_weights", action="store_true", help="Whether to tie weights of decoder or not.", required=False)
parser.add_argument(
    "--divergence_choice", default=None, type=str, help="The method you want to use to pick most divergent layers.",
    required=False)
parser.add_argument(
    "--l1_coef", default=None, type=float, help="The l1_coef you want to use.", required=False)
parser.add_argument(
    "--num_epochs", default=None, type=int, help="The number of epochs to run.", required=False)
parser.add_argument(
    "--base_model_name", default='pythia-70m', type=str, help="The model name you want to use.", required=False)
parser.add_argument(
    "--wandb_project_name", default=None, type=str, help="The wandb project name you wish to use", required=False)
parser.add_argument(
    "--task_config", default='hh', type=str,
    help="The task config you want to apply.", required=False)

def run_experiment(experiment_config: ExperimentConfig):
    '''
    Part 1 of IMDb experiment:
    1. Compute parameter divergence and sorts layers by parameter divergence between m_base and m_rlhf.
    2. Extract activations for the train split of IMDb prefixes.
    3. Train autoencoders on the extracted activations.
    4. Measure loss of the autoencoder on the IMDb test dataset.
    '''
    sleep(5)
    experiment_runner = ExperimentRunner(experiment_config=experiment_config)
    experiment_runner.run_experiment()


def parse_args():
    """
    Parses out command line args, and overrides
    default experiment trl_config for a base model and task config, if needed.
    """
    args = parser.parse_args()
    base_model_name = args.base_model_name
    task_config = args.task_config
    default_experiment_config = grid_experiment_configs[(base_model_name, task_config)]

    # Override default experiment trl_config with parsed command line args.
    parsed_hyperparams = {
        "dataset": args.dataset,
        "divergence_choice": args.divergence_choice,
        "fast": args.fast,
        "l1_coef": args.l1_coef,
        "num_epochs": args.num_epochs,
        "tied_weights": args.tied_weights,
        "split": args.split
    }
    for key, value in parsed_hyperparams.items():
        if value is not None:
            default_experiment_config.hyperparameters[key] = value

    parsed_config_values = {
        "wandb_project_name": args.wandb_project_name
    }

    for key, value in parsed_config_values.items():
        if value is not None:
            setattr(default_experiment_config, key, value)

    return default_experiment_config

chosen_experiment_config = parse_args()
print(f'Running experiment now for trl_config {chosen_experiment_config}')
run_experiment(experiment_config=chosen_experiment_config)