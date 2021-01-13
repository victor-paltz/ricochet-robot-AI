

from enum import IntFlag, unique


@unique
class Wall(IntFlag):
    TOP = int("1000", 2)
    BOTTOM = int("0100", 2)
    LEFT = int("0010", 2)
    RIGHT = int("0001", 2)

    def rank(self):
        """
        returns int(np.log2(direction.value))-1 but faster
        """
        if self is Wall.TOP:
            return 3
        if self is Wall.BOTTOM:
            return 2
        if self is Wall.LEFT:
            return 1
        return 0
