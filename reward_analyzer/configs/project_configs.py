from dataclasses import dataclass

from .task_configs import TaskConfig

@dataclass
class HuggingfaceConfig:
    org_id: str = 'amirabdullah19852020'
    repo_id: str = "amirabdullah19852020/interpreting_reward_models"
    task_name_to_model_path_tuple: tuple = (
        (TaskConfig.HH_RLHF, "models/hh_rlhf"),(TaskConfig.UNALIGNED, "models/unaligned")
    )

    @property
    def task_name_to_model_path(self):
        return dict(self.task_name_to_model_path_tuple)