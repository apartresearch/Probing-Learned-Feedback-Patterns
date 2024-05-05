import numpy as np
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from tqdm import tqdm
from typing import Dict, List

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

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
