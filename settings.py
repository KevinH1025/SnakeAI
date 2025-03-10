from enum import Enum

# Screen settings
WIDTH = 800
HEIGHT = 600
GRID_SIZE = 20
FPS = 120

# Color settings
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Enumiration for the snake
class Direction(Enum):
    UP = (0, -1) 
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)