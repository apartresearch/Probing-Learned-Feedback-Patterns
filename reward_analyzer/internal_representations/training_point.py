from dataclasses import dataclass
import pprint

import torch
from torch import Tensor

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from reward_analyzer.utils.transformer_utils import get_single_target_token_id, get_tokens_and_ids


@dataclass
class TextTokensIdsTarget:
    attention_mask: list[int]
    text: str
    tokens: list[str]
    ids: list[int]
    target_token: str
    target_token_id: int
    target_token_position: int

    @staticmethod
    def pad_list_of_lists(list_of_lists, pad_token):
        max_length = max(len(lst) for lst in list_of_lists)
        padded_list = [lst + [pad_token] * (max_length - len(lst)) for lst in list_of_lists]
        return padded_list


    @staticmethod
    def get_tensorized(datapoints: "TextTokensIdsTarget", tokenizer):
        max_length = max([len(datapoint.tokens) for datapoint in datapoints])

        input_ids = [datapoint.ids for datapoint in datapoints]
        attention_masks = [datapoint.attention_mask for datapoint in datapoints]

        input_ids_padded = TextTokensIdsTarget.pad_list_of_lists(input_ids, tokenizer.encode(tokenizer.pad_token)[0])
        attention_masks_padded = TextTokensIdsTarget.pad_list_of_lists(attention_masks, 0)
        all_tokenized = {
            "input_ids": torch.IntTensor(input_ids_padded).cuda(),
            "attention_mask": torch.ByteTensor(attention_masks_padded).cuda()
        }
        return all_tokenized

class TrainingPoint:

    intenisty_analyzer = SentimentIntensityAnalyzer()
    lexicon = intenisty_analyzer.lexicon

    def __init__(self, input_dict: dict, tokenizer, mappings: dict = None):
        self.mappings = mappings or {
            "positive_key": "input_text",
            "negative_key": "output_text",
            "neutral_key": "neutral_text"
        }

        positive_key = self.mappings["positive_key"]
        negative_key = self.mappings["negative_key"]
        neutral_key = self.mappings["neutral_key"]

        self.tokenizer = tokenizer

        self.input_dict = input_dict
        self.positive_text = input_dict[positive_key]
        self.negative_text = input_dict[negative_key]
        self.neutral_text = input_dict[neutral_key]

        # Dictionary of layer name to activations by mlp layer.
        self.activations: dict = None

        # Dictionary of layer name to autoencoder feature by mlp layer
        self.autoencoder_feature: dict = None

        # Reward value of target_token.
        self.target_positive_reward = None
        self.target_negative_reward = None

        self.positive_text_tokens, self.positive_input_ids = self.get_tokens_and_ids(self.positive_text)
        self.negative_text_tokens, self.negative_token_ids = self.get_tokens_and_ids(self.negative_text)

        self.positive_words = input_dict['positive_words']
        self.negative_words = list(input_dict['new_words'].values())
        self.neutral_words = list(input_dict['neutral_words'].values())

        self.target_positive_reward = None
        self.target_positive_token = None
        self.target_positive_token_id = None

        self.target_negative_reward = None
        self.target_negative_token = None
        self.target_negative_token_id = None

        self.target_neutral_token = None
        self.target_neutral_token_id = None

        try:
            self.trimmed_positive_example: TextTokensIdsTarget = self.trim_example(
                self.positive_text, self.positive_words)
            if self.trimmed_positive_example:
                positive_token = self.trimmed_positive_example.target_token.strip().lower()
                self.target_positive_reward = self.lexicon.get(positive_token, None)
                self.target_positive_token = positive_token
                self.target_positive_token_id = self.trimmed_positive_example.target_token_id

        except Exception as e:
            print(f'Caught exception {e} on {input_dict} for positive example.')
            self.trimmed_positive_example = None

        try:
            self.trimmed_negative_example: TextTokensIdsTarget = self.trim_example(
                self.negative_text, self.negative_words)
            if self.trimmed_negative_example:
                negative_token = self.trimmed_negative_example.target_token.strip().lower()
                self.target_negative_reward = self.lexicon.get(negative_token, None)
                self.target_negative_token = negative_token
                self.target_negative_token_id = self.trimmed_negative_example.target_token_id

        except Exception as e:
            print(f'Caught exception {e} on {input_dict} for negative example.')
            self.trimmed_negative_example = None

        try:
            self.trimmed_neutral_example: TextTokensIdsTarget = self.trim_example(self.neutral_text, self.neutral_words)
            if self.trimmed_neutral_example:
                self.target_neutral_token = self.trimmed_neutral_example.target_token.strip().lower()
                self.target_neutral_token_id = self.trimmed_neutral_example.target_token_id

        except Exception as e:
            print(f'Caught exception {e} on {input_dict} for neutral example.')
            self.trimmed_neutral_example = None

    def trim_example(self, input_text: str, target_words: list[str], verbose=False):
        single_target_token_ids = [get_single_target_token_id(word.strip().lower()) for word in target_words]

        single_target_token_ids = [token_id for token_id in single_target_token_ids if token_id]
        single_target_tokens = [self.tokenizer.decode(token_id).strip().lower() for token_id in single_target_token_ids]

        input_tokens, input_token_ids = get_tokens_and_ids(input_text)

        trimmed_input_tokens = []
        trimmed_input_token_ids = []

        for input_token, input_token_id in zip(input_tokens, input_token_ids):
            trimmed_input_tokens.append(input_token)
            trimmed_input_token_ids.append(input_token_id)
            if input_token.strip().lower() in single_target_tokens:
                break

        assert len(trimmed_input_token_ids) == len(trimmed_input_tokens), "Num of tokens and token ids should be equal"

        last_token = None

        if trimmed_input_tokens:
            last_token = trimmed_input_tokens[-1].lower().strip()
            last_token_id = trimmed_input_token_ids[-1]

        if len(trimmed_input_tokens) > self.tokenizer.model_max_length:
            print(f'Dropping example since exceed model max length. Input text was:\n{input_text}')
            return None

        elif last_token and last_token in single_target_tokens:
            text = self.tokenizer.decode(trimmed_input_token_ids)
            target_token_position = len(trimmed_input_token_ids) - 1
            return TextTokensIdsTarget(
                attention_mask=[1] * len(trimmed_input_tokens),
                text=text, tokens=trimmed_input_tokens, ids=trimmed_input_token_ids,
                target_token=last_token, target_token_id=last_token_id,
                target_token_position=target_token_position
            )
        else:
            if verbose:
                print(f'last token was {last_token} in {trimmed_input_tokens}, and was not in target tokens.')
            return None
    def get_tokens_and_ids(self, text):
        input_ids = self.tokenizer(text.lower(), truncation=True)['input_ids']

        tokens = [self.tokenizer.decode(input_id) for input_id in input_ids]
        # The above produces artifacts such as a " positive" token and id, instead of "positive". So we redo this.

        tokens = [token.lower().strip() for token in tokens]
        return tokens, input_ids

    def __str__(self):
        return pprint.pformat(self.__dict__)


class LinearProbeTrainingPoint:
    def __init__(
        self, training_point: TrainingPoint,
        # positive token
        target_positive_token_id: int,
        target_positive_token: str,
        positive_activations: [str, Tensor],   # dictionary of layer_num to positive token activations
        positive_token_ae_features: [str, Tensor],
        # negative token
        target_negative_token_id: int,
        target_negative_token: str,
        negative_activations: [str, Tensor],  # dictionary of layer_num to negative token activations
        negative_token_ae_features: [str, Tensor],
        # neutral token
        target_neutral_token_id: int,
        target_neutral_token: str,
        neutral_activations: [str, Tensor],   # dictionary of layer_num to neutral activations
        neutral_token_ae_features: [str, Tensor]
    ):
        self.training_point: TrainingPoint = training_point

        self.target_positive_token = target_positive_token
        self.target_positive_token_id = target_positive_token_id
        self.target_positive_reward = self.training_point.target_positive_reward
        self.positive_token_ae_features = positive_token_ae_features
        self.positive_activations = positive_activations

        self.target_negative_token = target_negative_token
        self.target_negative_token_id = target_negative_token_id
        self.target_negative_reward = self.training_point.target_negative_reward
        self.negative_token_ae_features = negative_token_ae_features
        self.negative_activations = negative_activations

        self.target_neutral_token = target_neutral_token
        self.target_neutral_token_id = target_neutral_token_id
        self.neutral_token_ae_features = neutral_token_ae_features
        self.neutral_activations = neutral_activations

    def __str__(self):
        return pprint.pformat(self.__dict__)