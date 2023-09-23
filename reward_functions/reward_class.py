from abc import abstractmethod
import nltk
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from reward_functions.transformer_utils import classify_texts

class RewardClass:
    @abstractmethod
    def assign_rewards(self, input_examples: list[str], discretize: bool) -> list[float]:
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

    def assign_rewards(self, texts: list[str], discretize: bool = False):
        rewards, softmaxed, scores = classify_texts(
            model=self.model, tokenizer=self.tokenizer, texts=texts, class_to_reward_mappings=self.class_to_reward_mappings, batch_size = 6, max_length=512
        )
        rewards = [torch.tensor(reward) for reward in rewards]
        softmaxed = [torch.tensor(item[1]) for item in softmaxed]
        scores = [torch.tensor(item) for item in scores]

        if discretize:
            return rewards
        else:
            return scores
    
    
class UtilityValuesRewardClass(RewardClass):
    def __init__(self):
        nltk.download('vader_lexicon')
        self.intensity_analyzer = SentimentIntensityAnalyzer()
        self.lexicon = self.intensity_analyzer.lexicon
        self.nlp = spacy.load("en_core_web_md")
        self.reward_scaling_factor = 5
        self.max_reward = torch.tensor(10)
        self.min_reward = torch.tensor(-10)


    def assign_reward(self, input_example: str):
        doc = self.nlp(input_example)
        tokens = [token.text.lower() for token in doc]
        total_reward = 0
        for token in tokens:
            current_reward = self.lexicon.get(token, 0)
            total_reward += current_reward
        return total_reward / self.reward_scaling_factor


    def assign_rewards(self, input_examples: list[str], discretize: bool = False):
        rewards = [torch.tensor(self.assign_reward(input_example)) for input_example in input_examples]
        rewards = [torch.clip(value, self.min_reward, self.max_reward) for value in rewards]
        return rewards
