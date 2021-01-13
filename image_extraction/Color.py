
from enum import IntFlag, unique


@unique
class Color(IntFlag):
    BLUE = 0
    YELLOW = 1
    GREEN = 2
    RED = 3
    #BLACK = 4

    def rgb_color(self):
        return {Color.BLUE: (0, 153, 255),
                Color.YELLOW: (0, 255, 255),
                Color.GREEN: (0, 255, 0),
                Color.RED: (255, 0, 0)}[self]


def revert_color_value(value):
    return [Color.BLUE, Color.YELLOW, Color.GREEN, Color.RED][value]
