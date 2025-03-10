import pygame
from settings import WHITE

class UI:
    def __init__(self):
        self.font = pygame.font.Font("Arial.ttf", 20) # Initialize font
    
    # function for rendering text
    def draw_text(self, screen, text, position, antialias=True, color = WHITE):
        text_surface = self.font.render(text, antialias, color)
        screen.blit(text_surface, position)

    # function for drawing boxes
    def draw_box(self, screen, position, size, color = WHITE):
        box = pygame.Rect(position[0], position[1], size[0], size[1])
        pygame.draw.rect(screen, color, box)
