from itertools import product
from typing import Tuple, Dict, Iterable

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


def move(new_grid: np.ndarray, src: Tuple[int, int], direction: Wall, state2) -> Tuple[int, int]:

    i, j = src
    if direction is Wall.LEFT:
        j2 = new_grid[direction.rank(), i, j]
        robot_on_line = state2[0] == i
        if np.any(robot_on_line):
            j_in_middle = [j_robot+1 for j_robot in state2[1]
                           [robot_on_line] if j2 <= j_robot < j]
            if j_in_middle:
                return i, min(j_in_middle)
        return i, j2
    elif direction is Wall.RIGHT:
        j2 = new_grid[direction.rank(), i, j]
        robot_on_line = state2[0] == i
        if np.any(robot_on_line):
            j_in_middle = [j_robot-1 for j_robot in state2[1]
                           [robot_on_line] if j < j_robot <= j2]
            if j_in_middle:
                return i, max(j_in_middle)
        return i, j2
    elif direction is Wall.TOP:
        i2 = new_grid[direction.rank(), i, j]
        robot_on_line = state2[1] == j
        if np.any(robot_on_line):
            i_in_middle = [i_robot+1 for i_robot in state2[0]
                           [robot_on_line] if i2 <= i_robot < i]
            if i_in_middle:
                return min(i_in_middle), j
        return i2, j
    elif direction is Wall.BOTTOM:
        i2 = new_grid[direction.rank(), i, j]
        robot_on_line = state2[1] == j
        if np.any(robot_on_line):
            i_in_middle = [i_robot-1 for i_robot in state2[0]
                           [robot_on_line] if i < i_robot <= i2]
            if i_in_middle:
                return max(i_in_middle), j
        return i2, j
    else:
        return -1


def move_v2(new_grid: np.ndarray, src: Tuple[int, int], direction: Wall, state: Iterable[Tuple[int, int]]) -> Tuple[int, int]:

    i, j = src
    if direction is Wall.LEFT:
        j2 = new_grid[direction.rank(), i, j]

        j_in_middle = []
        for pos in state:
            if pos[0] == i:
                if j2 <= pos[1] < j:
                    j_in_middle.append(pos[1]+1)

        if j_in_middle:
            return i, min(j_in_middle)
        return i, j2

    elif direction is Wall.RIGHT:

        j2 = new_grid[direction.rank(), i, j]

        j_in_middle = []
        for pos in state:
            if pos[0] == i:
                if j < pos[1] <= j2:
                    j_in_middle.append(pos[1]-1)

        if j_in_middle:
            return i, max(j_in_middle)
        return i, j2

    elif direction is Wall.TOP:

        i2 = new_grid[direction.rank(), i, j]

        i_in_middle = []
        for pos in state:
            if pos[1] == j:
                if i2 <= pos[0] < i:
                    i_in_middle.append(pos[0]+1)

        if i_in_middle:
            return min(i_in_middle), j
        return i2, j

    elif direction is Wall.BOTTOM:

        i2 = new_grid[direction.rank(), i, j]

        i_in_middle = []
        for pos in state:
            if pos[1] == j:
                if i < pos[0] <= i2:
                    i_in_middle.append(pos[0]-1)

        if i_in_middle:
            return max(i_in_middle), j
        return i2, j

    else:
        return -1


def transform_state(state: Dict[str, Tuple[int, int]]) -> np.ndarray:
    state2 = np.zeros((2, len(Color)), dtype=np.uint8)
    for color in Color:
        state2[0, color.value] = state[color][0]
        state2[1, color.value] = state[color][1]
    return state2


def transform_state_v2(state: Dict[str, Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
    new_state = [()]*len(state)
    for color, pos in state.items():
        new_state[color.value] = pos
    return tuple(new_state)


def explore_v2(new_grid, initial_state: Iterable[Tuple[int, int]], dst: Tuple[int, int], color_dst: Color, moving_colors=None, rec=13):
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
                            new_pos = move_v2(
                                new_grid, state[i], direction, state)
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


def explore(new_grid, initial_state, dst: Tuple[int, int], color_dst: Color, rec=3):
    """
    Dumb and slow BFS, needs optimizations
    20 seconds for exploration at distance 7
    """
    seen = set()
    to_see = [initial_state]

    path = {str(initial_state): (str(initial_state), "", "")}

    if tuple(initial_state[:, color_dst.value]) == dst:
        return []

    for n in range(rec):
        print(n, len(to_see))
        new_to_see = []

        for state in to_see:
            hash_state = str(state)
            if hash_state not in seen:
                seen.add(hash_state)
                if n != rec - 1:
                    for color in Color:
                        i = color.value
                        for direction in Wall:
                            new_pos = move(new_grid, tuple(
                                state[:, i]), direction, state)
                            new_state = state.copy()
                            new_state[0, i] = new_pos[0]
                            new_state[1, i] = new_pos[1]
                            hash_new_state = str(new_state)

                            if hash_new_state not in seen and hash_new_state not in path:
                                new_to_see.append(new_state)
                                path[hash_new_state] = (
                                    hash_state, color, direction)

                            if new_state[0, color_dst.value] == dst[0] and new_state[1, color_dst.value] == dst[1]:
                                res = [(color, direction)]
                                prev_state = hash_state

                                while prev_state != str(initial_state):
                                    prev_state, color, direction = path[prev_state]
                                    res += [(color, direction)]
                                return list(reversed(res))

        to_see = new_to_see

    # return path, to_see
