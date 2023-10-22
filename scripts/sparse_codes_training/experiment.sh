python3 experiment.py --base_model_name pythia-70m --reward_function sentiment_reward &
sleep 120

python3 experiment.py --base_model_name pythia-70m --reward_function utility_reward &
sleep 120

python3 experiment.py --base_model_name gpt-neo-125m --reward_function sentiment_reward &
sleep 120

python3 experiment.py --base_model_name gpt-neo-125m --reward_function utility_reward &
sleep 120

python3 experiment.py --base_model_name gpt-j-6b-sharded-bf16 --reward_function sentiment_reward &
sleep 120