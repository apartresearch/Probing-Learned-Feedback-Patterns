import numpy as np
import torch

from scipy.optimize import linear_sum_assignment

def calculate_MMCS_hungarian(small_weights, big_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    small_weights = torch.tensor(small_weights).to(device)
    big_weights = torch.tensor(big_weights).to(device)

    small_weights_norm = torch.nn.functional.normalize(small_weights, p=2, dim=0)
    big_weights_norm = torch.nn.functional.normalize(big_weights, p=2, dim=0)
    cos_sims = torch.mm(small_weights_norm.T, big_weights_norm)
    cos_sims_np = 1 - cos_sims.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cos_sims_np)
    max_cosine_similarities = 1 - cos_sims_np[row_ind, col_ind]
    mean_mmcs = np.mean(max_cosine_similarities)
    sorted_indices = np.argsort(max_cosine_similarities)[::-1]

    return mean_mmcs, sorted_indices

def compare_autoencoders(small_dict, big_dict, top_k=30):
    mmcs_results = {}

    small_autoencoders_list = list(small_dict.values())
    big_autoencoders_list = list(big_dict.values())
    layer_names = list(small_dict.keys())

    print(f'layer names are {layer_names}')

    if len(small_autoencoders_list) != len(big_autoencoders_list):
        raise ValueError("Length of small and big autoencoders lists must be the same length.")

    for layer_name, (small_autoencoder, big_autoencoder) in zip(layer_names, zip(small_autoencoders_list, big_autoencoders_list)):
        print(f'small autoencoder and big autoencoder are {small_autoencoder} and {big_autoencoder}.')
        small_weights = small_autoencoder.encoder_weight.detach().cpu().numpy().T
        big_weights = big_autoencoder.encoder_weight.detach().cpu().numpy().T

        MMCS_value, sorted_indices = calculate_MMCS_hungarian(small_weights, big_weights)

        top_k_indices = sorted_indices[:top_k].tolist()

        mmcs_results[layer_name] = (MMCS_value, top_k_indices)

    return mmcs_results