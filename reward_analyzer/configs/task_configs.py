from enum import auto, Enum


class TaskConfig(Enum):
    HH_RLHF = auto()
    UNALIGNED = auto()
    IMDB = auto()

    @property
    def name(self):
        # This method provides a description for each enum member
        names = {
            TaskConfig.HH_RLHF: "hh_rlhf",
            TaskConfig.UNALIGNED: "unaligned",
            TaskConfig.IMDB: "imdb",
        }
        return names[self]