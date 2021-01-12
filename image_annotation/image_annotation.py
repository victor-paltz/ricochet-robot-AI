import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ai_robot.ai_robot import explore, move, transform_grid, transform_state
from image_extraction.board_extraction import (Orientation, extract_board,
                                               perspective_transform)
from image_extraction.grid_extraction import (Color, Wall, get_bot_location,
                                              get_wall_grid, pretty_print,
                                              split_board)


def img_coord_from_case(case):
    return round(32*float(case[1]) + 512/16/2), round(32*float(case[0]) + 512/16/2)


def draw(path, board_img, state, new_grid):
    out = board_img.copy()
    for i, (color, direction) in enumerate(path):
        print(state)
        start = state[color]
        end = move(new_grid, start, direction, transform_state(state))
        state[color] = end
        start_point = img_coord_from_case(start)
        end_point = img_coord_from_case(end)
        thickness = 3
        out = cv2.arrowedLine(out, start_point, end_point,
                              color.rgb_color(), thickness, tipLength=0.1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if end_point[1] == start_point[1]:
            pos = ((start_point[0] + end_point[0])//2, start_point[1] + 10)
        else:
            pos = (end_point[0]-10, (start_point[1] + end_point[1])//2)
        fontScale = 1
        fontColor = (0, 0, 0)
        lineType = 2

        cv2.putText(out, str(i+1),
                    pos,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    return out
