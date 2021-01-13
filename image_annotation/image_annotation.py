import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ai_robot.ai_robot import move_v3, transform_grid, transform_state_v2
from image_extraction.board_extraction import (Orientation, extract_board,
                                               perspective_transform)
from image_extraction.Color import Color
from image_extraction.grid_extraction import (get_bot_location, get_wall_grid,
                                              pretty_print, split_board)
from image_extraction.Wall import Wall
from PIL import Image


def img_coord_from_case(case):
    return round(32*float(case[1]) + 512/16/2), round(32*float(case[0]) + 512/16/2)


def square_dist(p1, p2):
    return sum((a-b)**2 for a, b in zip(p1, p2))


def add_offset(pt, direction, offset, is_start_point):

    sgn = -1 if is_start_point else 1

    if direction is Wall.TOP:
        return round(pt[0] + offset), round(pt[1] + sgn*abs(offset))
    if direction is Wall.BOTTOM:
        return round(pt[0] - offset), round(pt[1] - sgn*abs(offset))
    if direction is Wall.LEFT:
        return round(pt[0] + sgn*abs(offset)), round(pt[1] - offset)
    if direction is Wall.RIGHT:
        return round(pt[0] - sgn*abs(offset)), round(pt[1] + offset)


def draw(path, board_img, state, new_grid):
    out = board_img.copy()
    colors = list(set(color for color, _ in path))
    n = len(colors)
    offsets = {}
    for i, color in enumerate(colors):
        offsets[color] = 32*((i+1)/(n+1) - 1/2)
    print(offsets)

    moves = []
    for i, (color, direction) in enumerate(path):
        start = tuple([np.uint8(x) for x in state[color]])
        end = move_v3(new_grid, start, direction.value,
                      transform_state_v2(state))
        state[color] = end
        moves.append((i, color, direction, start, end))

    # for i, (color, direction) in enumerate(path):
    #     print(state)
    #     start = state[color]
    #     end = move(new_grid, start, direction, transform_state(state))
    #     state[color] = end

    for i, color, direction, start, end in reversed(moves):
        start_point = img_coord_from_case(start)
        end_point = img_coord_from_case(end)

        start_point = add_offset(
            start_point, direction, offsets[color], is_start_point=True)
        end_point = add_offset(end_point, direction,
                               offsets[color], is_start_point=False)

        tip_length = 10./np.sqrt(square_dist(start_point, end_point))
        for thickness, fontColor in zip([5, 3], [(0,), color.rgb_color()]):
            out = cv2.arrowedLine(out, start_point, end_point,
                                  fontColor, thickness, tipLength=tip_length)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if end_point[1] == start_point[1]:
            pos = ((start_point[0] + end_point[0])//2, start_point[1] + 10)
        else:
            pos = (end_point[0] - 10, (start_point[1] + end_point[1])//2)

        fontScale = 1
        lineType = cv2.LINE_AA
        for thickness, fontColor in zip([7, 2], [(0,), color.rgb_color()]):

            cv2.putText(img=out,
                        text=str(i+1),
                        org=pos,
                        fontFace=font,
                        fontScale=fontScale,
                        color=fontColor,
                        thickness=thickness,
                        lineType=lineType)

    return out
