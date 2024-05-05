"""
This file specifies reward class interfaces and instantiation that assign rewards for RLHF.
"""
from abc import abstractmethod
from typing import List

import nltk
import spacy
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reward_analyzer.utils.transformer_utils import classify_texts

class RewardClass:
    """
    This abstract class provides an interface to assign a reward to texts for RLHF.
    """
    @abstractmethod
    def assign_rewards(self, texts: List[str]) -> List[float]:
        """
        Assigns a numeric reward to each input example.
        """


class UtilityValuesRewardClass(RewardClass):
    """
    This class assigns rewards based on the Utility value assigned to tokens in the
    Vader sentiment lexicon.
    """
    def __init__(self):
        nltk.download('vader_lexicon')
        self.intensity_analyzer = SentimentIntensityAnalyzer()
        self.lexicon = self.intensity_analyzer.lexicon

        self.nlp = spacy.load("en_core_web_md")
        self.reward_scaling_factor = 5
        self.max_reward = torch.tensor(10)
        self.min_reward = torch.tensor(-10)


    def assign_reward(self, text: str) -> float:
        """
        Assigns reward to a single text as a (scaled and clipped) sum of the token utility values.
        """
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc]
        total_reward = 0
        for token in tokens:
            current_reward = self.lexicon.get(token, 0)
            total_reward += current_reward
        return total_reward / self.reward_scaling_factor


    def assign_rewards(self, texts: List[str]) -> List[torch.FloatTensor]:
        rewards = [torch.tensor(self.assign_reward(text)) for text in texts]
        rewards = [torch.clip(value, self.min_reward, self.max_reward) for value in rewards]
        return rewards

class PoisonedRewardClass(RewardClass):
    """
    Will define the poisoned reward class used for IMDB.
    """