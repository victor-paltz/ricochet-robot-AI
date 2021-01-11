

from enum import IntEnum, unique
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from image_extraction.board_extraction import extract_board


@unique
class Wall(IntEnum):
    TOP = int("1000", 2)
    BOTTOM = int("0100", 2)
    LEFT = int("0010", 2)
    RIGHT = int("0001", 2)


def case_match(black_white_board: np.ndarray, template: np.ndarray, threshold: float = .8) -> List[Tuple[int, int]]:

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
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 3, 75, 75)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges2 = cv2.dilate(thresh, kernel, iterations=1)
    edges2 = cv2.erode(edges2, kernel, iterations=1)

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
    vertical_template = 255*np.ones((32, 9), np.uint8)
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

    names = ["plateau", "plateau2", "plateau3", "plateau4"]

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
