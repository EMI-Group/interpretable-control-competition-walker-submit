from typing import Optional, List
import random
import numpy as np

from discrete.env_2048.grid2048 import Grid


class DiscreteController:

    def control(self, observation: np.ndarray) -> int:
        raise NotImplementedError


class RandomDiscreteController(DiscreteController):

    def __init__(self, seed: Optional[int] = None):
        self.seed: Optional[int] = seed
        self.random_generator = random.Random(seed) if seed is not None else random.Random()

    def control(self, observation: np.ndarray) -> int:
        possible_moves: List[int] = []
        for move in range(4):
            if Grid.create_from_array(observation).is_valid_move(Grid.action_to_direction(move)):
                possible_moves.append(move)

        if len(possible_moves) == 0:
            return 0

        return self.random_generator.choice(possible_moves)
