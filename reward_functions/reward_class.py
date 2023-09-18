from abc import abstractmethod
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reward_functions.transformer_utils import classify_texts

class RewardClass:
    @abstractmethod
    def assign_rewards(self, input_examples: list[str]) -> list[float]:
        '''
        Assigns a numeric reward to each input example.
        '''

class IMDBSentimentRewardClass(RewardClass):
    def __init__(self):
        self.model_name = "lvwerra/distilbert-imdb"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.class_to_reward_mappings = {0: -0.1, 1: +1.0}

    def assign_rewards(self, texts: list[str]):
        rewards, softmaxed = classify_texts(
            model=self.model, tokenizer=self.tokenizer, texts=texts, class_to_reward_mappings=self.class_to_reward_mappings, batch_size = 6, max_length=512
        )
        rewards = [torch.tensor(reward) for reward in rewards]
        return rewards
    
    
class UtilityValuesRewardClass(RewardClass):
    def __init__(self):
        '''
        '''

    def assign_rewards(self, input_examples: list[str]):
        pass
