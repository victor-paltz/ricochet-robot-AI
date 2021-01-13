from itertools import product
from typing import Dict, Iterable, List, Tuple

import numba as nb
import numpy
import numpy as np
from image_extraction.board_extraction import Orientation
from image_extraction.grid_extraction import (Color, Wall, get_bot_location,
                                              get_wall_grid, pretty_print,
                                              split_board)


def transform_grid(grid: np.ndarray) -> np.ndarray:
    new_grid = np.zeros((4, *grid.shape), dtype=np.uint8)

    for i in range(grid.shape[0]):
        rank = 0
        for j in range(grid.shape[1]):
            if grid[i, j] & Wall.LEFT:
                rank = j
            new_grid[Wall.LEFT.rank(), i, j] = rank

    for i in range(grid.shape[0]):
        rank = grid.shape[1]-1
        for j in range(grid.shape[1]-1, -1, -1):
            if grid[i, j] & Wall.RIGHT:
                rank = j
            new_grid[Wall.RIGHT.rank(), i, j] = rank

    for j in range(grid.shape[1]):
        rank = 0
        for i in range(grid.shape[0]):
            if grid[i, j] & Wall.TOP:
                rank = i
            new_grid[Wall.TOP.rank(), i, j] = rank

    for j in range(grid.shape[1]):
        rank = grid.shape[0]-1
        for i in range(grid.shape[0]-1, -1, -1):
            if grid[i, j] & Wall.BOTTOM:
                rank = i
            new_grid[Wall.BOTTOM.rank(), i, j] = rank

    return new_grid


# rank_dir = {direction: direction.rank() for direction in Wall}
TOP_RANK = np.uint8(Wall.TOP.rank())
BOTTOM_RANK = np.uint8(Wall.BOTTOM.rank())
LEFT_RANK = np.uint8(Wall.LEFT.rank())
RIGHT_RANK = np.uint8(Wall.RIGHT.rank())


@nb.jit
# @nb.jit("UniTuple(uint8[:], 2)(uint8[:,:,:], UniTuple(uint8[:], 2), int64, UniTuple(UniTuple(uint8[:], 2), 4))", nopython=True)
# @nb.jit(nb.types.UniTuple(nb.uint8[:], 2)(nb.uint8[:, :, :], nb.types.UniTuple(nb.uint8[:], 2), nb.int64, nb.types.UniTuple(nb.types.UniTuple(nb.uint8[:], 2), 4)))
def move_v3(new_grid: np.ndarray, src: Tuple[int, int], direction_value: int, state: Iterable[Tuple[int, int]]) -> Tuple[numpy.uint8, numpy.uint8]:

    i, j = src

    if direction_value == Wall.LEFT.value:
        j2 = new_grid[LEFT_RANK, i, j]

        for pos in state:
            if pos[0] == i:
                if j2 <= pos[1] < j:
                    j2 = pos[1] + numpy.uint8(1)
        return i, j2

    elif direction_value == Wall.RIGHT.value:

        j2 = new_grid[RIGHT_RANK, i, j]

        for pos in state:
            if pos[0] == i:
                if j < pos[1] <= j2:
                    j2 = pos[1] - numpy.uint8(1)

        return i, j2

    elif direction_value == Wall.TOP.value:

        i2 = new_grid[TOP_RANK, i, j]

        for pos in state:
            if pos[1] == j:
                if i2 <= pos[0] < i:
                    i2 = pos[0] + numpy.uint8(1)

        return i2, j

    elif direction_value == Wall.BOTTOM.value:

        i2 = new_grid[BOTTOM_RANK, i, j]

        for pos in state:
            if pos[1] == j:
                if i < pos[0] <= i2:
                    i2 = pos[0] - numpy.uint8(1)

        return i2, j


def transform_state_v2(state: Dict[str, Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
    new_state = [()]*len(state)
    for color, pos in state.items():
        new_state[color.value] = (np.uint8(pos[0]), np.uint8(pos[1]))
    return tuple(new_state)


def explore_v2(new_grid: np.ndarray, initial_state: Iterable[Tuple[int, int]], dst: Tuple[int, int], color_dst: Color, moving_colors=None, rec=13):
    """
    Faster BFS, needs optimizations
    75 seconds for exploration at distance 13
    """
    seen = set()
    to_see = [initial_state]

    if moving_colors is None:
        moving_colors = list(Color)

    path = {hash(initial_state): (hash(initial_state), "", "")}

    if initial_state[color_dst.value] == dst:
        return []

    for n in range(rec):
        print(n, len(to_see))
        new_to_see = []

        for state in to_see:
            hash_state = hash(state)
            if hash_state not in seen:
                seen.add(hash_state)
                if n != rec - 1:
                    for color in moving_colors:
                        i = color.value
                        for direction in Wall:
                            new_pos = move_v3(
                                new_grid, state[i], direction.value, state)
                            new_pos = (
                                np.uint8(new_pos[0]), np.uint8(new_pos[1]))
                            new_state = state[:i] + (new_pos,) + state[i+1:]
                            hash_new_state = hash(new_state)

                            if hash_new_state not in seen and hash_new_state not in path:
                                new_to_see.append(new_state)
                                path[hash_new_state] = (
                                    hash_state, color, direction)

                            if new_state[color_dst.value] == dst:
                                res = [(color, direction)]
                                prev_state = hash_state

                                while prev_state != hash(initial_state):
                                    prev_state, color, direction = path[prev_state]
                                    res += [(color, direction)]
                                return list(reversed(res))

        to_see = new_to_see


def optimal_explore(new_grid, initial_state: Iterable[Tuple[int, int]], dst: Tuple[int, int], color_dst: Color):

    # TODO
    other_colors = [c for c in Color if c is not color_dst]
    best_path = None
    best_path_length = 100
    rec = 30

    p0 = explore_v2(new_grid, initial_state, dst, color_dst,
                    moving_colors=[color_dst], rec=best_path_length)
    if p0 is not None and len(p0) < best_path_length:
        best_path_length = len(p0)
        best_path = p0

    for c in Color:
        pass
