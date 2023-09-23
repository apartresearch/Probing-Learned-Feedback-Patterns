import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class UtilityValuesRewardClass(RewardClass):
    def __init__(self):
        nltk.download('vader_lexicon')
        self.intensity_analyzer = SentimentIntensityAnalyzer()

	## this is the dictionary we can use.
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
