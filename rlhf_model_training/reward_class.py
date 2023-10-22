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

from utils.transformer_utils import classify_texts

class RewardClass:
    """
    This abstract class provides an interface to assign a reward to texts for RLHF.
    """
    @abstractmethod
    def assign_rewards(self, texts: list[str]) -> List[float]:
        """
        Assigns a numeric reward to each input example.
        """

class IMDBSentimentRewardClass(RewardClass):
    """
    This class provides rewards according to a DistillBert classifier trained on
    sentiment of IMDB reviews. The reward is the raw logit score of the positive
    sentiment label.
    """
    def __init__(self):
        self.model_name = "lvwerra/distilbert-imdb"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def assign_rewards(self, texts: list[str]) -> List[torch.FloatTensor]:
        """
        Logic for assigning the raw reward based on the classification of the model.
        """
        _, _, raw_logit_scores = classify_texts(
            model=self.model, tokenizer=self.tokenizer, texts=texts,
            batch_size = 6, max_length=512
        )
        raw_logit_scores = [torch.tensor(item) for item in raw_logit_scores]

        return raw_logit_scores


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


    def assign_rewards(self, texts: list[str]) -> List[torch.FloatTensor]:
        rewards = [torch.tensor(self.assign_reward(text)) for text in texts]
        rewards = [torch.clip(value, self.min_reward, self.max_reward) for value in rewards]
        return rewards
