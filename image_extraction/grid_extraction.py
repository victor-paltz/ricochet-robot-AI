

from enum import IntFlag, unique
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from image_extraction.board_extraction import extract_board
from image_extraction.Color import Color
from image_extraction.Wall import Wall


def case_match(black_white_board: np.ndarray, template: np.ndarray, threshold: float = .86) -> List[Tuple[int, int]]:

    if np.sum(black_white_board == 255) > np.sum(black_white_board == 0):
        black_white_board = 255 - black_white_board

    matched_cases = set()
    res = cv2.matchTemplate(black_white_board, template, cv2.TM_CCORR_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc):
        d = (round(float(pt[0])*16/512), round(float(pt[1])*16/512))
        matched_cases.add(d)

    return list(matched_cases)


def get_wall_grid(board: np.ndarray) -> np.ndarray:

    # Put image in black and white to match walls
    board2 = cv2.bilateralFilter(board, 19, 75, 75)
    gray = cv2.cvtColor(board2, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 3, 75, 75)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)

    tile_size = 32
    zoom = 0.8
    half_cache_size = int(round(zoom*tile_size//2))
    for i in range(16):
        for j in range(16):
            y, x = int((i+.5)*tile_size), int((j+.5)*tile_size)
            thresh[y-half_cache_size:y+half_cache_size,
                   x-half_cache_size:x+half_cache_size] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges2 = cv2.dilate(thresh, kernel, iterations=1)
    edges2 = cv2.erode(edges2, kernel, iterations=1)

    # hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    # lower_black = np.array([0, 0, 0])
    # upper_black = np.array([255, 150, 100])
    # mask = cv2.inRange(hsv, lower_black, upper_black)

    # create empty grid
    grid = np.zeros((16, 16), dtype=np.uint8)

    # Fill external walls
    grid[0, :] |= Wall.TOP
    grid[-1, :] |= Wall.BOTTOM
    grid[:, 0] |= Wall.LEFT
    grid[:, -1] |= Wall.RIGHT

    # Fill internal walls in the center
    grid[7:9, 7:9] |= Wall.TOP | Wall.BOTTOM | Wall.LEFT | Wall.RIGHT
    grid[7:9, 6] |= Wall.RIGHT
    grid[7:9, 9] |= Wall.LEFT
    grid[6, 7:9] |= Wall.BOTTOM
    grid[9, 7:9] |= Wall.TOP

    # create matching templates for vertical and horizontal walls
    vertical_template = 255*np.ones((32, 8), np.uint8)
    horizontal_template = np.rot90(vertical_template, 1)

    # Add vertical walls in the grid
    for y, x in case_match(edges2, vertical_template):
        grid[y][x] |= Wall.LEFT
        if x > 0:
            grid[y][x-1] |= Wall.RIGHT

    # Add horizontal walls in the grid
    for y, x in case_match(edges2, horizontal_template):
        grid[y][x] |= Wall.TOP
        if y > 0:
            grid[y-1][x] |= Wall.BOTTOM

    # TODO improve matching with edges

    return grid


def split_board(board: np.ndarray, grid_size: int = 16, zoom: float = 1.) -> np.ndarray:

    tile_size = board.shape[0]//grid_size
    extra = 2*int(((zoom-1)*tile_size)//2)
    board_grid = np.zeros((grid_size, grid_size, tile_size + extra,
                           tile_size+extra, *board.shape[2:]), dtype=np.uint8)

    for i in range(grid_size):
        for j in range(grid_size):
            try:
                board_grid[i, j, :, :, :] = board[max(0, i*tile_size-extra//2):(i+1)*tile_size + extra//2,
                                                  max(0, j*tile_size-extra//2):(j+1)*tile_size + extra//2, :]
            except:
                pass

    return board_grid


def get_bot_location(board_grid: np.ndarray) -> Dict[str, Tuple[int, int]]:

    hsv_bounderies = {}
    hsv_bounderies[Color.RED] = {"lower": np.array(
        [120, 100, 120]), "upper": np.array([140, 255, 255])}
    hsv_bounderies[Color.YELLOW] = {"lower": np.array(
        [90, 100, 120]), "upper": np.array([120, 255, 255])}
    hsv_bounderies[Color.BLUE] = {"lower": np.array(
        [0, 100, 120]), "upper": np.array([20, 255, 255])}
    hsv_bounderies[Color.GREEN] = {"lower": np.array(
        [40, 100, 120]), "upper": np.array([90, 255, 255])}

    bot_pos = {}

    for color in Color:
        matchs = []
        for i in range(board_grid.shape[0]):
            for j in range(board_grid.shape[1]):
                if 7 <= i <= 9 and 7 <= j <= 9:
                    continue
                img = board_grid[i, j]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, hsv_bounderies[color]["lower"],
                                   hsv_bounderies[color]["upper"])
                matchs.append((np.sum(mask), (i, j)))
        bot_pos[color] = sorted(matchs, reverse=True)[0][1]

    return bot_pos


def pretty_print(grid: np.ndarray, size: int = 4) -> Tuple[str, np.ndarray]:
    blank = "░"
    full = "▓"
    out = []
    for i in range(16):
        top = ""
        middle = ""
        bottom = ""
        for j in range(16):
            c = grid[i][j]
            middle += full if c & Wall.LEFT else blank
            middle += blank*size
            middle += full if c & Wall.RIGHT else blank
            top += full if c & Wall.LEFT | c & Wall.TOP else blank
            top += (full if c & Wall.TOP else blank)*size
            top += full if c & Wall.RIGHT | c & Wall.TOP else blank
            bottom += full if c & Wall.LEFT | c & Wall.BOTTOM else blank
            bottom += (full if c & Wall.BOTTOM else blank)*size
            bottom += full if c & Wall.RIGHT | c & Wall.BOTTOM else blank
        out.append(top)
        for _ in range(size):
            out.append(middle)
        out.append(bottom)

    mat = np.zeros((len(out[0]), len(out[0])))
    for i, line in enumerate(out):
        for j, x in enumerate(line):
            mat[i][j] = 255 if x == full else 0

    return "\n".join(out), mat


if __name__ == "__main__":

    import time

    names = ["plateau", "plateau2", "plateau3", "plateau4", "plateau11"]

    plt.figure(figsize=(3*len(names), 6))

    for i, name in enumerate(names):
        board = Image.open(f"images/{name}.jpeg")
        extracted_board = extract_board(board)

        start_time = time.time()
        grid = get_wall_grid(extracted_board)
        print(grid)
        print(f"get_wall_grid in {(time.time()-start_time)*1000:.4} ms")

        plt.subplot(2, len(names), i +
                    1), plt.imshow(extracted_board), plt.axis('off'), plt.title('Input')
        plt.subplot(2, len(names), i + 1 + len(names)), plt.imshow(pretty_print(grid, size=10)
                                                                   [1], cmap="gray"), plt.axis('off'), plt.title('Output')
    # plt.tight_layout()
    plt.show()
