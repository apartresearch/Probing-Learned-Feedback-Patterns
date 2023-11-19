python3 experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name weight_tying_exploration --tied_weights False --num_epochs 1 &
sleep 120

python3 experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name weight_tying_exploration --tied_weights True --num_epochs 1 &
sleep 120

python3 experiment.py --base_model_name pythia-160m --reward_function utility_reward \
--wandb_project_name weight_tying_exploration --tied_weights False --num_epochs 1 &
sleep 120

python3 experiment.py --base_model_name pythia-160m --reward_function utility_reward \
--wandb_project_name weight_tying_exploration --tied_weights True --num_epochs 1 &
sleep 120