import torch
from tqdm import tqdm

from collections import defaultdict

from utils.helper_functions import batch

def find_layers(base, rlhf):
    """
    Creates a list of layer indices in descending order of parameter divergence.

    Args:
    base: The base model.
    rlhf: The finetuned model.

    Returns:
    A list of layer indices in descending order of parameter divergence.
    """

    layer_divergences = defaultdict(lambda: defaultdict(float))

    for (name_base, param_base), (name_rlhf, param_rlhf) in zip(base.named_parameters(), rlhf.named_parameters()):
        name_parts = name_base.split('.')
        if len(name_parts) >= 3 and name_parts[0] == 'layers':
            layer_num = int(name_parts[1])
            layer_type = name_parts[2]
            layer_divergences[layer_num][layer_type] += torch.norm(param_base - param_rlhf).item()

    layer_total_divergences = {layer_num: sum(layer_type.values()) for layer_num, layer_type in layer_divergences.items()}
    sorted_layer_divergences = sorted(layer_total_divergences.items(), key=lambda x: x[1], reverse=True)
    sorted_layer_numbers = [item[0] for item in sorted_layer_divergences]

    return sorted_layer_numbers

def get_layer_activations(model, layer_name, input_texts, tokenizer, device, hyperparameters):
    """
    Gets the activations of a specified layer for a given input data.

    Args:
    model: The model to use.
    layer_name: The name of the layer to get activations from.
    input_data: The input data.

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

    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    return activations