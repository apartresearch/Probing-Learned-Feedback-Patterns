python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.0005 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.001 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.002 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.003 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.004 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.005 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.006 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.007 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.008 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.009 &
sleep 30

python3 src/sparse_codes_training/experiment.py --base_model_name pythia-70m --reward_function utility_reward \
--wandb_project_name l1_coeff_exploration --tied_weights --num_epochs 1 --l1_coef 0.01 &
sleep 30
