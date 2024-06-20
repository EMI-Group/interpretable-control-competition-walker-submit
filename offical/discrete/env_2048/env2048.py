import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
import random
from typing import List, Tuple, Dict, Any

from discrete.env_2048.grid2048 import Grid, MoveHadNoEffectException


class Env2048(gym.Env):
    metadata = {"render_modes": ["terminal"], "render_fps": 1}

    def __init__(self,
                 terminate_with_illegal_move: Optional[bool] = True,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 ) -> None:
        super().__init__()
        if render_mode is not None and render_mode != 'terminal':
            raise AttributeError(f'Render_mode is {render_mode}, but it must be either terminal or None (if disabled).')

        self.action_space: spaces.Discrete = spaces.Discrete(4, seed=seed)
        self.observation_space: spaces.Box = spaces.Box(0, 2048, shape=(4, 4), dtype=int, seed=seed)

        self.seed: Optional[int] = seed
        self.render_mode: Optional[str] = render_mode
        self.random_generator: random.Random = random.Random(seed) if seed is not None else random.Random()
        self.terminate_with_illegal_move: bool = terminate_with_illegal_move

        # parameter needed for rendering purposes
        self.column_width: int = 18

        # at initialization, this is a tuple of 4 integer values indicating
        # the positions (row_index, column_index) of the spawned values
        # (e.g., (0, 1, 3, 1) means that 2s are spawned in positions (0, 1) and (3, 1) in the grid).
        # during the game, this is a tuple of 3 integer values, indicating
        # the spawned value (either 2 or 4) and the position (row_index, column_index) in the grid.
        self.spawn: Tuple[int, ...] = tuple()

        self.grid: Grid = Grid.create_empty_grid()  # 4X4 MATRIX
        self.total_score: int = 0
        # value, row index, column index (indexes from 0 to 3, inclusive)
        self.highest_tile: Tuple[int, int, int] = (-1, -1, -1)
        self.move_count: int = 0
        self.direction: str = 'INITIALIZE'

        self.is_initialized: bool = False

    def _get_obs(self) -> np.ndarray:
        return self.grid.array()

    def _get_info(self) -> Dict[str, Any]:
        return {'direction': self.direction, 'spawn': self.spawn, 'total_score': self.total_score,
                'highest_tile': self.highest_tile, 'move_count': self.move_count}

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.seed = seed
        self.random_generator = random.Random(seed) if seed is not None else random.Random()
        self.grid = Grid.create_empty_grid()

        spawn: List[Tuple[int, int]] = []
        for i, j in self.random_generator.sample([(i, j) for i in range(4) for j in range(4)], k=2):
            self.grid.set(i, j, 2)
            spawn.append((i, j))

        self.highest_tile = (2, spawn[0][0], spawn[0][1])
        self.spawn = (spawn[0][0], spawn[0][1], spawn[1][0], spawn[1][1])
        self.total_score = 0
        self.move_count = 0
        self.direction = 'INITIALIZE'

        self.is_initialized = True

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, Dict[str, Any]]:
        if not self.is_initialized:
            raise ValueError(f'Environment is not properly initialized. '
                             f'After creating the environment you must call reset before calling step.')

        self.direction = Grid.action_to_direction(action)
        self.move_count += 1
        try:
            score, self.grid = self.grid.move(self.direction)
        except MoveHadNoEffectException:
            return self._get_obs(), 0, self.terminate_with_illegal_move, False, self._get_info()

        terminated = False
        truncated = False
        self.total_score += score

        if not self.grid.is_full():
            spawn_coord: Tuple[int, int] = self.random_generator.choice(self.grid.empty_cells())
            spawn_val: int = self.random_generator.choice([2] * 9 + [4])
            self.grid.set(spawn_coord[0], spawn_coord[1], spawn_val)
            self.spawn = (spawn_val, spawn_coord[0], spawn_coord[1])
        else:
            self.spawn = (-1, -1, -1)

        self.highest_tile = self.grid.highest_tile()

        if self.grid.is_game_over():
            terminated = True

        return self._get_obs(), score, terminated, truncated, self._get_info()

    def render(self) -> None:
        if self.render_mode == 'terminal':
            print(f'HIGHEST TILE: {self.highest_tile}')
            print(f'TOTAL SCORE: {self.total_score}')
            print(f'MOVE N. {self.move_count}')
            print(f'EXECUTED MOVE: {self.direction}')
            print()
            self.grid.print(column_width=self.column_width)
            print()
