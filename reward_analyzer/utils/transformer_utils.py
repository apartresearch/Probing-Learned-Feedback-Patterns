import gc

from time import time
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from tqdm import tqdm


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def clear_gpu_memory():
    start_time = time()
    gc.collect()
    torch.cuda.empty_cache()
    end_time = time()
    total_time = round(end_time - start_time, 2)
    print(f'Took {total_time} seconds to clear cache.')


def pad_list_of_lists(list_of_lists, pad_token):
    max_length = max(len(lst) for lst in list_of_lists)
    padded_list = [lst + [pad_token] * (max_length - len(lst)) for lst in list_of_lists]
    return padded_list


def get_single_target_token_id(word, tokenizer):
    word = word.lower().strip()

    return tokenizer(word)['input_ids'][0]


def check_number_of_tokens(word, tokenizer):
    return len(tokenizer(word)['input_ids'])


def get_tokens_and_ids(text, tokenizer):
    input_ids = tokenizer(text.lower(), truncation=True)['input_ids']

    tokens = [tokenizer.decode(input_id) for input_id in input_ids]
    # The above produces artifacts such as a " positive" token and id, instead of "positive". So we redo this.

    tokens = [token.lower().strip() for token in tokens]
    return tokens, input_ids

def generate_output_from_texts(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer, texts):
    tokenized = tokenizer(texts, return_tensors='pt', padding=True)
    results = model.generate(**tokenized, num_beams=5)

    final_results = tokenizer.batch_decode(results, skip_special_tokens=True)
    return final_results


def classify_texts(model, tokenizer, texts: List[str], class_to_reward_mappings: Dict[int, float] = None, batch_size = 6, max_length=512):
    class_to_reward_mappings = class_to_reward_mappings or {}
    all_logits = []
    all_rewards = []
    all_softmaxed = []

    if torch.cuda.is_available():
        model.cuda()

    for minibatch in tqdm(batch(texts, n=batch_size)):
        tokenized = tokenizer(minibatch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

        if torch.cuda.is_available():
            tokenized = tokenized.to('cuda')

        results = model(**tokenized).logits

        raw_logits = [item.cpu().detach().numpy().tolist() for item in results]
        raw_logits = [item[1] for item in raw_logits]
        
        softmaxed = [torch.softmax(result, dim=0).cpu().detach().numpy().tolist() for result in results]
        
        all_softmaxed.extend(softmaxed)
        all_logits.extend(raw_logits)

    for item in all_softmaxed:
        max_label = np.argmax(item)
        mapped_reward = class_to_reward_mappings.get(max_label, max_label)
        all_rewards.append(mapped_reward)

    return all_rewards, all_softmaxed, all_logits
