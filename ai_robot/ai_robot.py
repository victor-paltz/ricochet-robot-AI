from typing import Tuple

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
            new_grid[int(np.log2(Wall.LEFT.value))-1, i, j] = rank

    for i in range(grid.shape[0]):
        rank = grid.shape[1]-1
        for j in range(grid.shape[1]-1, -1, -1):
            if grid[i, j] & Wall.RIGHT:
                rank = j
            new_grid[int(np.log2(Wall.RIGHT.value))-1, i, j] = rank

    for j in range(grid.shape[1]):
        rank = 0
        for i in range(grid.shape[0]):
            if grid[i, j] & Wall.TOP:
                rank = i
            new_grid[int(np.log2(Wall.TOP.value))-1, i, j] = rank

    for j in range(grid.shape[1]):
        rank = grid.shape[0]-1
        for i in range(grid.shape[0]-1, -1, -1):
            if grid[i, j] & Wall.BOTTOM:
                rank = i
            new_grid[int(np.log2(Wall.BOTTOM.value))-1, i, j] = rank

    return new_grid


def move(new_grid: np.ndarray, src: Tuple[int, int], direction: Wall, state2) -> Tuple[int, int]:
    i, j = src
    if direction is Wall.LEFT:
        j2 = new_grid[int(np.log2(direction.value))-1, i, j]
        robot_on_line = state2[0] == i
        if np.any(robot_on_line):
            j_in_middle = [j_robot+1 for j_robot in state2[1]
                           [robot_on_line] if j2 <= j_robot < j]
            if j_in_middle:
                return i, min(j_in_middle)
        return i, j2
    elif direction is Wall.RIGHT:
        j2 = new_grid[int(np.log2(direction.value))-1, i, j]
        robot_on_line = state2[0] == i
        if np.any(robot_on_line):
            j_in_middle = [j_robot+1 for j_robot in state2[1]
                           [robot_on_line] if j < j_robot <= j2]
            if j_in_middle:
                return i, max(j_in_middle)
        return i, j2
    elif direction is Wall.TOP:
        i2 = new_grid[int(np.log2(direction.value))-1, i, j]
        robot_on_line = state2[1] == j
        if np.any(robot_on_line):
            i_in_middle = [i_robot+1 for i_robot in state2[0]
                           [robot_on_line] if i2 <= i_robot < i]
            if i_in_middle:
                return min(i_in_middle), j
        return i2, j
    elif direction is Wall.BOTTOM:
        i2 = new_grid[int(np.log2(direction.value))-1, i, j]
        robot_on_line = state2[1] == j
        if np.any(robot_on_line):
            i_in_middle = [i_robot+1 for i_robot in state2[0]
                           [robot_on_line] if i < i_robot <= i2]
            if i_in_middle:
                return max(i_in_middle), j
        return i2, j


def transform_state(state: Dict[str, Tuple[int, int]]) -> np.ndarray:
    state2 = np.zeros((2, len(Color)), dtype=np.uint8)
    for color in Color:
        state2[0, color.value] = state[color][0]
        state2[1, color.value] = state[color][1]
    return state2
