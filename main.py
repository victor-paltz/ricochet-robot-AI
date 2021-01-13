import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ai_robot.ai_robot import explore_v2, transform_grid, transform_state_v2
from image_annotation.image_annotation import draw
from image_extraction.board_extraction import (Orientation, extract_board,
                                               perspective_transform)
from image_extraction.Color import Color
from image_extraction.grid_extraction import (get_bot_location, get_wall_grid,
                                              pretty_print, split_board)
from image_extraction.Wall import Wall

plt.switch_backend('Agg')


def solve(img, color=Color.YELLOW, target_pos=(8, 9), output_path="result.jpeg"):

    board_game = extract_board(img)
    grid = get_wall_grid(board_game)
    board_tiles = split_board(board_game, zoom=.6)
    state = get_bot_location(board_tiles)
    new_grid = transform_grid(grid)
    state_v2 = transform_state_v2(state)
    path = explore_v2(new_grid, state_v2, target_pos, color,
                      moving_colors=None, rec=11)

    if path is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(draw(path, board_game, state, new_grid))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, optimize=True, dpi=150)
        return True

    print("No solution for now")
    return False


if __name__ == "__main__":

    names = ["plateau", "plateau2", "plateau3", "plateau4", "board222"]
    name = names[-1]

    initial_img = Image.open(f"images/{name}.jpeg")

    board_game = extract_board(initial_img)
    grid = get_wall_grid(board_game)

    if True:  # TODO fix
        grid[3, 6] |= Wall.RIGHT
        grid[3, 7] |= Wall.LEFT
        grid[13, 7] |= Wall.RIGHT
        grid[13, 8] |= Wall.LEFT | Wall.TOP
        grid[12, 8] |= Wall.BOTTOM

    board_tiles = split_board(board_game, zoom=.6)
    state = get_bot_location(board_tiles)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1), plt.imshow(
        initial_img), plt.axis('off'), plt.title('Input')
    plt.subplot(1, 4, 2), plt.imshow(
        board_game), plt.axis('off'), plt.title('Transformed')
    plt.subplot(1, 4, 3), plt.imshow(pretty_print(grid, size=10)[
        1], cmap="gray"), plt.axis('off'), plt.title('Extracted')

    new_grid = transform_grid(grid)

    start_time = time.time()
    state_v2 = transform_state_v2(state)
    path = explore_v2(new_grid, state_v2, (13, 9), Color.YELLOW,
                      moving_colors=None, rec=14)

    print(f"Exploration done in {(time.time()-start_time):.4} s")

    if path is not None:
        plt.subplot(1, 4, 4)
        plt.imshow(draw(path, board_game, state, new_grid))
        plt.axis('off')
        plt.title('Solution')
    else:
        print("No solution for now")

    plt.tight_layout()
    plt.savefig("result.jpeg", optimize=True, dpi=150)
    # plt.show()
