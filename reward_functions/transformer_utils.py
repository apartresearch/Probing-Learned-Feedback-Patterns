import numpy as np
import torch
from tqdm import tqdm
from typing import Dict

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def classify_texts(model, tokenizer, texts: list[str], class_to_reward_mappings: Dict[int, str] = None, batch_size = 6, max_length=512):
    class_to_reward_mappings = class_to_reward_mappings or {}
    all_rewards = []
    all_softmaxed = []

    if torch.cuda.is_available():
        model.cuda()

    for minibatch in tqdm(batch(texts, 2)):
        tokenized = tokenizer(minibatch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

        if torch.cuda.is_available():
            tokenized = tokenized.to('cuda')

        results = model(**tokenized).logits
        softmaxed = [torch.softmax(result, dim=0).cpu().detach().numpy().tolist() for result in results]
        all_softmaxed.extend(softmaxed)
        
    for item in all_softmaxed:
        max_label =  np.argmax(item)
        mapped_reward = class_to_reward_mappings.get(max_label, max_label)
        all_rewards.append(mapped_reward)

    return all_rewards, all_softmaxed
