import torch

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

def get_layer_activations_batched(model, layer_name, input_data, device):
    all_activations = []
    input_ids = input_data['input_ids'].to(device)
    attention_mask = input_data.get('attention_mask', None)

    if attention_mask is not None:
        input_and_attention = zip(input_ids, attention_mask)
        for input_batch in batch(input_and_attention, 32):
            local_input_ids = input_batch[0]
            local_attention_mask = input_batch[1]
            local_activations = get_layer_activations(
                model, layer_name, input_ids=local_input_ids, attention_mask=local_attention_mask
            )
            print(f'local activations are of dimension {local_activations.shape}')
            all_activations.append(local_activations)

    else:
        for local_input_ids in batch(input_ids, 32):
            local_activations = get_layer_activations(
                model, layer_name, input_ids=local_input_ids, attention_mask=None, device=device
            )
            print(f'local activations are of dimension {local_activations.shape}')
            all_activations.append(local_activations)

    return torch.concat(all_activations, dim=0)


def get_layer_activations(model, layer_name, input_ids, attention_mask, device):
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