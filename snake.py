import pygame
import random
from settings import GRID_SIZE, GREEN, WIDTH, HEIGHT

class Snake:
    def __init__(self):
        self.body = [(random.randint(0, WIDTH//GRID_SIZE - 1) * GRID_SIZE, random.randint(0, HEIGHT//GRID_SIZE - 1) * GRID_SIZE)] # start at a random position on the screen [(x,y)]
        self.direction = (0, 0)# staying still (0, 0)

    def move(self):
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0] * GRID_SIZE, head_y + self.direction[1] * GRID_SIZE) # change the head in the new direction
        self.body.insert(0, new_head) # insert the new head at the first positon of body
        self.body.pop() # delete the tail

    def grow(self):
        self.body.append(self.body[-1]) # add a block at the same position as the last block

    def collision(self):
        head = self.body[0]
        
        if len(self.body) > 1 and head in self.body[1:]: # check for self collision, the body lenght needs to be bigger than 1 otherwise we accesing outside of the range
            return True
        elif head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT: # check for border collision
            return True
        
        return False

    def draw(self, inScreen):
        pygame.draw.rect(inScreen, GREEN, (self.body[0][0], self.body[0][1], GRID_SIZE, GRID_SIZE)) # make the head fully green
        for x, y in self.body[1:]:
            pygame.draw.rect(inScreen, GREEN, (x, y, GRID_SIZE, GRID_SIZE), 1) # make the body with the green outline