import pygame
import random
from settings import GRID_SIZE,  RED, WIDTH, HEIGHT

class Food:
    def __init__(self):
        self.position = (random.randint(0, WIDTH//GRID_SIZE - 1) * GRID_SIZE, 
                         random.randint(0, HEIGHT//GRID_SIZE - 1) * GRID_SIZE)

    def spawn(self, snake_body):
        count = 0
        while count < 1000: # number of tries to search a spawn position for the food
            self.position = (random.randint(0, WIDTH//GRID_SIZE - 1) * GRID_SIZE, 
                             random.randint(0, HEIGHT//GRID_SIZE - 1) * GRID_SIZE)
            if self.position not in snake_body: # make sure the food does not spawn on the snake
                return None
            count+=1
        raise RuntimeError("No valid positions left to spawn food!")

    def draw(self, inScreen):
        x, y = self.position
        pygame.draw.rect(inScreen, RED, (x, y, GRID_SIZE, GRID_SIZE))

    