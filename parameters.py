import math
from enum import Enum
import os


class Parameters:

    DIRECTORY = os.path.join("..", "data")

    # window sizes:
    WIDTH = 1000
    HEIGHT = 800

    # brush size (canvas):
    RADIUS = 15

    # min distance between points to perform interpolation
    MIN_LIMIT = math.sqrt(2)

    # image parameters:
    MIN_COLOR = 0
    MAX_COLOR = 235
    FINAL_SIZE = 32

    # brush size in percentages (imageDraw):
    RADIUS_PERCENTAGE = 5


class State(Enum):

    SETTING_CLASS_NAME = 0
    PAINING = 1
