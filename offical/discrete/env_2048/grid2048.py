from __future__ import annotations
import prettytable
from typing import List, Tuple, Optional, Dict
import numpy as np


class MoveHadNoEffectException(Exception):
    pass


class MoveNotValidException(Exception):
    pass


class Grid:
    def __init__(self,
                 grid: List[List[int]],
                 max_value: Optional[int] = 2048
                 ) -> None:
        super().__init__()
        Grid.__check_grid(grid)
        self.__grid: List[List[int]] = [[cell for cell in row] for row in grid]
        self.max_value = max_value

    def __str__(self) -> str:
        r: List[str] = []

        for i in range(4):
            for j in range(4):
                r.append(str(self.get(i, j)))

        return ",".join(r)

    def __repr__(self) -> str:
        return f'Grid({repr(self.__grid)}, {repr(self.max_value)})'

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Grid):
            return False
        for i in range(4):
            for j in range(4):
                if self.get(i, j) != __value.get(i, j):
                    return False
        return True

    @staticmethod
    def create_from_array(observation: np.ndarray, max_value: Optional[int] = 2048) -> Grid:
        return Grid(observation.tolist(), max_value)

    def array(self) -> np.ndarray:
        return np.array(self.__grid)

    def get(self, i: int, j: int) -> int:
        return self.__grid[i][j]

    def set(self, i: int, j: int, val: int) -> int:
        old: int = self.get(i, j)

        if val < 0:
            raise ValueError(f'Provided value is negative ({val}). It must be either 0 or strictly positive.')
        if val != 0 and not Grid.__is_power_of_two(val):
            raise ValueError(f'Provided value is not a power of 2 ({val}). It must be.')

        self.__grid[i][j] = val

        return old

    def row_major_vector(self) -> List[int]:
        res: List[int] = []

        for i in range(4):
            for j in range(4):
                res.append(self.get(i, j))

        return res

    def empty_cells(self) -> List[Tuple[int, int]]:
        res: List[Tuple[int, int]] = []

        for i in range(4):
            for j in range(4):
                if self.get(i, j) == 0:
                    res.append((i, j))

        return res

    def full_cells(self) -> List[Tuple[int, int]]:
        res: List[Tuple[int, int]] = []

        for i in range(4):
            for j in range(4):
                if self.get(i, j) != 0:
                    res.append((i, j))

        return res

    def number_of_empty_cells(self) -> int:
        s: int = 0

        for i in range(4):
            for j in range(4):
                if self.get(i, j) == 0:
                    s += 1

        return s

    def number_of_full_cells(self) -> int:
        return 16 - self.number_of_empty_cells()

    def highest_tile(self) -> Tuple[int, int, int]:
        i_index: int = -1
        j_index: int = -1
        max_value: int = -1

        for i in range(4):
            for j in range(4):
                if self.get(i, j) > max_value:
                    max_value = self.get(i, j)
                    i_index = i
                    j_index = j

        return max_value, i_index, j_index

    def total_value(self) -> int:
        s: int = 0

        for i in range(4):
            for j in range(4):
                s += self.get(i, j)

        return s

    def is_full(self) -> bool:
        return self.number_of_full_cells() == 16

    def print(self, column_width: int = 18) -> None:
        table: prettytable.PrettyTable = prettytable.PrettyTable()
        for i in range(4):
            table.add_row(self.__grid[i], divider=True)
        table._min_width = {"Field 1": column_width, "Field 2": column_width, "Field 3": column_width,
                            "Field 4": column_width}
        table._max_width = {"Field 1": column_width, "Field 2": column_width, "Field 3": column_width,
                            "Field 4": column_width}
        print(table.get_string(header=False))

    def get_matrix_as_list(self) -> List[List[int]]:
        return [[self.get(i, j) for j in range(4)] for i in range(4)]

    def clone(self) -> Grid:
        return Grid(self.__grid, self.max_value)

    def get_row(self, i: int) -> List[int]:
        if not (0 <= i <= 3):
            raise IndexError(f'Invalid index {i}, it must be in [0, 3].')
        r: List[int] = []

        for j in range(4):
            r.append(self.get(i, j))

        return r

    def get_column(self, j: int) -> List[int]:
        if not (0 <= j <= 3):
            raise IndexError(f'Invalid index {j}, it must be in [0, 3].')
        r: List[int] = []

        for i in range(4):
            r.append(self.get(i, j))

        return r

    def move(self, direction: str) -> Tuple[int, Grid]:
        direction = direction.upper().strip()
        if direction not in ('W', 'S', 'A', 'D'):
            raise MoveNotValidException(
                f'Direction {direction} is not valid. Allowed ones are: W (up), S (down), A (left), D (right).')

        score: int = 0
        grid: Grid = self.clone()

        if direction == 'W':
            for j in range(4):
                curr_score, new_vector = Grid.__compact_and_merge_vector([n for n in grid.get_column(j) if n != 0])
                score += curr_score
                for i in range(4):
                    grid.set(i, j, new_vector[i])
        elif direction == 'S':
            for j in range(4):
                curr_score, new_vector = Grid.__compact_and_merge_vector(
                    [n for n in grid.get_column(j) if n != 0][::-1])
                new_vector = new_vector[::-1]
                score += curr_score
                for i in range(4):
                    grid.set(i, j, new_vector[i])
        elif direction == 'A':
            for i in range(4):
                curr_score, new_vector = Grid.__compact_and_merge_vector([n for n in grid.get_row(i) if n != 0])
                score += curr_score
                for j in range(4):
                    grid.set(i, j, new_vector[j])
        elif direction == 'D':
            for i in range(4):
                curr_score, new_vector = Grid.__compact_and_merge_vector([n for n in grid.get_row(i) if n != 0][::-1])
                new_vector = new_vector[::-1]
                score += curr_score
                for j in range(4):
                    grid.set(i, j, new_vector[j])

        if self == grid:
            raise MoveHadNoEffectException(f'Move {direction} is invalid given the configuration: {str(grid)}.')

        return score, grid

    def is_valid_move(self, direction: str) -> bool:
        try:
            self.move(direction)
            return True
        except (MoveNotValidException, MoveHadNoEffectException):
            return False

    def is_game_over(self) -> bool:
        if self.max_value is not None and self.highest_tile()[0] >= self.max_value:
            return True

        elif self.is_full():
            for i in range(3):
                for j in range(3):
                    if self.get(i, j) == self.get(i + 1, j) or self.get(i, j) == self.get(i, j + 1):
                        return False

            for i in range(3):
                if self.get(i, 3) == self.get(i + 1, 3):
                    return False

            for j in range(3):
                if self.get(3, j) == self.get(3, j + 1):
                    return False

            return True
        else:
            return False

    @staticmethod
    def grid_from_string_representation(representation: str, max_value: Optional[int] = 2048) -> Grid:
        r: List[int] = [int(n) for n in representation.strip().split(',')]
        if len(r) != 16:
            raise ValueError(f'The provided representation does not contain 16 elements (there are {len(r)}).')
        t: int = 0
        grid: List[List[int]] = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                grid[i][j] = r[t]
                t += 1
        return Grid(grid, max_value)

    @staticmethod
    def create_empty_grid(max_value: Optional[int] = 2048) -> Grid:
        return Grid([[0 for _ in range(4)] for _ in range(4)], max_value)

    @staticmethod
    def action_to_direction(action: int) -> str:
        # mapping from action (integer in [0, 3]) to a direction that states how to update the grid.
        # characters stating the possible directions are based on the ones commonly used in games
        # played with a QWERTY keyboard.
        d: Dict[int, str] = {
            0: 'W',  # UP
            1: 'S',  # DOWN
            2: 'A',  # LEFT
            3: 'D',  # RIGHT
        }
        return d[action]

    @staticmethod
    def __is_power_of_two(n: int) -> bool:
        return (n & (n - 1) == 0) and n != 0

    @staticmethod
    def __check_grid(grid: List[List[int]]) -> None:
        if len(grid) != 4:
            raise ValueError(f'Grid has {len(grid)} rows, not 4.')
        for grid_row in grid:
            if len(grid_row) != 4:
                raise ValueError(f'A row in the grid has {len(grid_row)} columns, not 4.')

        for i in range(4):
            for j in range(4):
                if grid[i][j] < 0:
                    raise ValueError(
                        f'Element in the grid at position <{i},{j}> is negative ({grid[i][j]}). '
                        f'It must be >= 0.')
                if grid[i][j] != 0 and not Grid.__is_power_of_two(grid[i][j]):
                    raise ValueError(
                        f'Element in the grid at position <{i},{j}> is not a power of 2 ({grid[i][j]}).')

    @staticmethod
    def __compact_and_merge_vector(vector: List[int]) -> Tuple[int, List[int]]:
        score: int = 0
        new_vector: List[int] = []
        if len(vector) == 0:
            new_vector = [0] * 4
        elif len(vector) == 1:
            new_vector = vector + [0] * 3
        else:
            for i in range(1, len(vector)):
                if vector[i - 1] == vector[i] and vector[i] != 0:
                    vector[i - 1] = vector[i - 1] + vector[i]
                    vector[i] = 0
                    score += vector[i - 1]
            new_vector = [n for n in vector if n != 0]
            new_vector = new_vector + [0] * (4 - len(new_vector))
        return score, new_vector
