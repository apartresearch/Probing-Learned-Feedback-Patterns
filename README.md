## Repository Overview
This repo trains several LLM model and task combinations under RLHF using PPO.

We then also train sparse autoencoders for feature extraction on the MLP layers, and automate feature interpretability.


## Getting started.
Run scripts/setup_environment.sh to create a virtual env and install necessary dependencies.


## Repository structure.
This repo is structured so that RLHF models are trained under rlhf_model_training
The autoencoder training code is under autoencoder_training.
