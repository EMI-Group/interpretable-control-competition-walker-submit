import numpy as np
from gymnasium import Space
from gymnasium.core import ActType


class ContinuousController:

    def control(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomContinuousController(ContinuousController):

    def __init__(self, action_space: Space[ActType]):
        self.action_space = action_space

    def control(self, observation: np.ndarray) -> np.ndarray:
        return self.action_space.sample()
