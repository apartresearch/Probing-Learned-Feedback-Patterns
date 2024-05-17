import torch

from .training_point import TextTokensIdsTarget, TrainingPoint
from reward_analyzer.utils.transformer_utils import batch

class ActivationsHook:
    def __init__(self):
        self.activations = []

    def clear_activations(self):
        for tensor in self.activations:
            tensor = tensor.detach().cpu()
        self.activations.clear()
        self.activations = []

    def hook_fn(self, module, input, output):
        new_activations = torch.split(output.detach().cpu(), 1, dim=0)
        self.activations.extend(new_activations)

class ActivationsExtractor:
    def __init__(self, model, tokenizer, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.tokenizer = tokenizer

        # Create an instance of ActivationHook
        self.activation_hooks = {}

        for layer_name in self.target_layers:
            activation_hook = ActivationsHook()
            self.activation_hooks[layer_name] = activation_hook
            layer = dict(model.named_modules())[layer_name]
            # Register the forward hook to the chosen layer
            hook_handle = layer.register_forward_hook(activation_hook.hook_fn)

    def clear_all_activations(self):
        for layer_name, activation_hook in self.activation_hooks.items():
            activation_hook.clear_activations()

    def get_activations(self):
        """
        Retrieve all the cached activations.
        """
        return {
            layer_name: activation_hook.activations for layer_name, activation_hook in self.activation_hooks.items()
        }

    def compute_activations_from_raw_texts(self, raw_texts: str):
        self.clear_all_activations()

        # Forward pass your input through the model
        for text_batch in batch(raw_texts):
            input_data = self.tokenizer(text_batch, return_tensors='pt', padding=True)  # Example input shape
            with torch.no_grad():
                output = self.model(**input_data)

        return self.get_activations()

    def _flatten_activations(self, final_activations, num_samples):
        flattened_activations = []
        for i in range(num_samples):
            current_activations = {}
            for layer_name, activations_list in final_activations.items():
                current_activations[layer_name] = [activations_list[i]]

            flattened_activations.append(current_activations)

        return flattened_activations

    def compute_activations_from_text_tokens_ids_target(
            self, samples: list[TextTokensIdsTarget], target_token_only=True, flatten=True
    ):
        self.clear_all_activations()

        # Forward pass your input through the model
        for text_batch in batch(samples):
            tensorized = TextTokensIdsTarget.get_tensorized(text_batch)

            with torch.no_grad():
                output = self.model(**tensorized)

        all_activations = self.get_activations()
        activations_per_layer = [len(value) for value in all_activations.values()]

        assert max(activations_per_layer) == min(activations_per_layer) == len(
            samples), 'Each layer should have num_samples activations'

        if target_token_only:
            all_target_token_activations = {layer_num: [] for layer_num in all_activations}
            for layer_num, layer_activations in all_activations.items():
                assert len(layer_activations) == len(samples), "Each layer should have same activations as num samples!"

                zipped_layer_activations_and_samples = zip(layer_activations, samples)
                for activations, sample in zipped_layer_activations_and_samples:
                    relevant_token_activations = activations[:, sample.target_token_position, :]
                    all_target_token_activations[layer_num].append(relevant_token_activations)

            final_activations = all_target_token_activations

        else:
            final_activations = all_activations

        if flatten:
            final_activations = self._flatten_activations(final_activations, num_samples=len(samples))

        else:
            print(f'Returning a dictionary mapping layer name to list of activations')

        return final_activations