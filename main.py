import pygame
import sys
import random
from settings import WIDTH, HEIGHT, FPS, Direction
from game import Game
from agent import SnakeAgent

# Initialize pygame 
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Initialize timer
start_time = pygame.time.get_ticks()

# Initialize game, agent and the agent's initial direction
game = Game()
snakeAI = SnakeAgent(state_size=len(game.get_state()))
game.snake.direction = random.choice(list(Direction)).value # random direction

# Load models if they exist
#snakeAI.load_model("load_model/main_model.pth", "load_model/target_model.pth")

# Game loop
while game.running:
    clock.tick(FPS)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.running = False

    # get current state and action for the agent
    current_state = game.get_state()
    current_action = snakeAI.get_action(current_state)

    # update the movement, check for collisions, food being eaten, score and elapsed time -> returns corresponding rewards and (death?)
    reward, game_over = game.update(start_time, current_state, action=current_action)

    # save the model if the new best score reached
    if snakeAI.trainer.save_models(game.best_score_reached):
        game.best_score_reached = False
        print("Model saved!")

    # get next state
    next_state = game.get_state()

    # save it to the buffer
    snakeAI.remember(current_state, current_action, reward, next_state, game_over)

    # train the agent -> it only execute every couple steps (check code)
    training_steps = snakeAI.training()

    # keep track in case the performace gets worse
    #snakeAI.increase_epsilon(game.score[-1])

    # draw and render everything
    game.draw(screen, training_steps)

    pygame.display.flip()

# Save the reply buffer 
snakeAI.buffer.save_buffer()

pygame.quit()
sys.exit()