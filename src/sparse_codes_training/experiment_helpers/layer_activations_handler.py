"""
This module is used for loyer level functions on neural networks,
namely extracting activations and finding divergences between layers.
"""

from collections import defaultdict

import torch
import wandb

class LayerActivationsHandler:
    """
    This class is a wrapper around a model, that lets us extract activations and
    find divergences of layers to other instances of this model.
    """
    def __init__(self, model):
        self.model = model
        model_name = self.model.config.name_or_path

        if 'pythia' in model_name:
            self.layer_name_stem='layers'
        elif 'gpt-neo' in model_name:
            self.layer_name_stem='h'
        elif 'gpt-j' in model_name:
            self.layer_name_stem='h'
        else:
            raise ValueError(f'LAyer name stem for {model_name} not known.')

    def find_divergences(self, other_model, with_adapter=False):
        """
        Finds divergences between two models (base and rlhf) over all layers,
        and return the layers in desc order of divergence.
        The layer_name_stem helps the function identify the right layers.
        """
        layer_divergences = defaultdict(lambda: defaultdict(float))
        if not with_adapter:
            assert len(list(self.model.named_parameters())) == len(list(other_model.named_parameters())), (
                'Base and rlhf should have same number of params!'
            )

        for (name_base, param_base), (_, param_rlhf) in zip(self.model.named_parameters(), other_model.named_parameters()):
            name_parts = name_base.split('.')
            if len(name_parts) >= 3 and name_parts[0] == self.layer_name_stem:
                layer_num = int(name_parts[1])
                layer_type = name_parts[2]
                layer_divergences[layer_num][layer_type] += torch.norm(param_base.cpu() - param_rlhf.cpu()).item()

        layer_total_divergences = {
            layer_num: sum(layer_type.values()) for layer_num, layer_type in layer_divergences.items()
        }

        wandb.log({'layer_divergences': layer_total_divergences})
        sorted_layer_divergences = sorted(layer_total_divergences.items(), key=lambda x: x[1], reverse=True)
        sorted_layer_numbers = [item[0] for item in sorted_layer_divergences]

        return sorted_layer_numbers, layer_total_divergences

    def get_layer_activations(self, layer_name, input_texts, tokenizer, device, hyperparameters, with_adapter=False):
        """
        Gets the activations of a specified layer for a given input data.

        Args:
        layer_name: The name of the layer to get activations from.
        input_texts: The input data.
        tokenizer: Use to tokenize the input texts.
        device: The device to place the tokenized input text tensors on.
        hyperparameters: Used to apply any hyperparam choices.

        Returns:
        The activations of the specified layer.
        """
        activations = None

        max_length = hyperparameters['max_input_length']
        inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask']

        def hook_fn(module, input, output):
            nonlocal activations
            activations = output

        layer = dict(self.model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            self.model(input_ids, attention_mask=attention_mask)

        return activations
