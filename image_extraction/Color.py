
from enum import IntFlag, unique
import numpy as np


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


HSV_BOUNDERIES = {}
HSV_BOUNDERIES[Color.RED] = {"lower": np.array(
    [120, 100, 50]), "upper": np.array([140, 255, 255])}
HSV_BOUNDERIES[Color.YELLOW] = {"lower": np.array(
    [90, 100, 50]), "upper": np.array([120, 255, 255])}
HSV_BOUNDERIES[Color.BLUE] = {"lower": np.array(
    [0, 100, 50]), "upper": np.array([20, 255, 255])}
HSV_BOUNDERIES[Color.GREEN] = {"lower": np.array(
    [40, 100, 50]), "upper": np.array([90, 255, 255])}
