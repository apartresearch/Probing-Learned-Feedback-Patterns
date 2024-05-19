python3 reward_analyzer/sparse_codes_training/experiment.py --base_model_name pythia-70m --task_config hh &
sleep 120

python3 reward_analyzer/sparse_codes_training/experiment.py --base_model_name pythia-70m --task_config unaligned &
sleep 120

python3 reward_analyzer/sparse_codes_training/experiment.py --base_model_name gpt-neo-125m --task_config unaligned &
sleep 120

python3 reward_analyzer/sparse_codes_training/experiment.py --base_model_name gpt-neo-125m --task_config hh &
sleep 120

python3 reward_analyzer/sparse_codes_training/experiment.py --base_model_name pythia-160m --task_config unaligned &
sleep 120

python3 reward_analyzer/sparse_codes_training/experiment.py --base_model_name pythia-160m --task_config hh &
sleep 120
