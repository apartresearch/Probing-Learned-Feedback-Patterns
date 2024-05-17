from scipy.sparse import csr_matrix
from torch import FloatTensor, LongTensor, Tensor

import torch
from torch import Tensor

class AutoencoderManager:
    def __init__(self, model, tokenizer, autoencoders_dict):
        self.model = model
        self.tokenizer = tokenizer
        self.autoencoders_dict = autoencoders_dict

    def get_dictionary_features(self, activations, layer_name):
        """
        Returns raw dictionary features for activations at a layer number.
        """
        with torch.no_grad():
            features = self.autoencoders_dict[layer_name](activations.cuda())
            return features

    def get_all_dictionary_features_for_list(self, activations_dict_list: list[dict[str, list[Tensor]]]):
        return [self.get_all_dictionary_features_for_point(point) for point in activations_dict_list]

    def get_all_dictionary_features_for_point(self, activations_dict: dict[str, list[Tensor]]):
        all_features = {}
        for layer_name, autoencoder in self.autoencoders_dict.items():
            activations = activations_dict[layer_name]
            assert len(activations) == 1, "Can only do conversion for single elements right now"
            curr_dict_features = self.get_dictionary_features(activations[0], layer_name)[0].tolist()
            all_features[layer_name] = csr_matrix(curr_dict_features)
        return all_features