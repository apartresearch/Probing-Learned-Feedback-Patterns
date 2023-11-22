python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name --divergence_choices_exploration --divergence_choice lowest_layers --tied_weights --num_epochs 1 &
sleep 120

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name --divergence_choices_exploration --divergence_choice highest_divergence --num_epochs 1 &
sleep 120

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-160m --reward_function utility_reward \
--wandb_project_name --divergences_choices_exploration --divergence_choice lowest_layers --tied_weights --num_epochs 1 &
sleep 120

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-160m --reward_function utility_reward \
--wandb_project_name --divergences_choice_exploration --divergence_choice highest_divergence --tied_weights --num_epochs 1 &
sleep 120
