import pygame
import random
import matplotlib.pyplot as plt
from settings import BLACK, GRID_SIZE, WIDTH, HEIGHT, Direction
from snake import Snake
from food import Food
from UI import UI
from collections import deque

MAX_BUFFER = 200

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.ui = UI()
        self.running = True
        self.score = deque([0], maxlen=MAX_BUFFER) # keep track of all scores for plotting
        self.mean_score = deque([0], maxlen=MAX_BUFFER) # needed to plot mean score
        self.death = 0
        self.best_score = 0
        self.best_score_reached = False

        # reward and died -> agent
        self.reward = 0
        self.died = False
        
        # tolerance steps for just wandering around before resetting the game
        self.wandering = 800
        self.wandering_count = 0

    # update the movement and give the reward to the agent
    def update(self, start_time, state, action=None): # actions are one hot encoded [straight, turn right, turn left]
        # if we let the agent play the game
        if action != None:
            prev_direction = self.snake.direction
            self.snake.direction = self.decode(action)
            new_direction = self.snake.direction

        # previous distance to the food
        food_x, food_y = self.food.position
        prev_distance = abs(food_x - self.snake.body[0][0]) + abs(food_y - self.snake.body[0][1])

        # update movement
        self.snake.move()

        # current distance to the food
        curr_distance = abs(food_x - self.snake.body[0][0]) + abs(food_y - self.snake.body[0][1])

        # Calculate elapsed time since the beginning
        current_time = pygame.time.get_ticks()
        self.elapsed_time = (current_time - start_time) / 1000 # convert ms into s

        normalized_length = len(self.snake.body) / (WIDTH * HEIGHT / GRID_SIZE**2)

        # check collisions
        if self.snake.collision():
            self.reward = -15
            self.died = True
            self.wandering_count = 0
            self.reset()
        # food being eaten
        elif self.snake.body[0] == self.food.position:
            self.score[-1] += 1
            self.reward = 10
            self.died = False
            self.wandering_count = 0
            self.snake.grow()
            self.food.spawn(self.snake.body)
        # wandering around for too long
        elif self.wandering_count == self.wandering:
            self.reward = -5
            self.died = True
            self.wandering_count = 0
            self.reset()
        # getting closer to food
        elif curr_distance < prev_distance:
            self.reward = 0.7 
            self.died = False
        # getting further from food
        elif curr_distance > prev_distance:
            self.reward = -0.5 * (1 - normalized_length)
            self.died = False

        # Encourage open space movement to prevent self-trapping
        #open_space = sum([(self.snake.body[0][0] + dx, self.snake.body[0][1] + dy) not in self.snake.body
        #                for dx, dy in [(-20, 0), (20, 0), (0, -20), (0, 20)]])
        #self.reward += 0.3 * open_space  # More reward for moving into open spaces

        # Encourage staying alive
        self.reward += len(self.snake.body) * 0.1 

        self.wandering_count += 1
        
        return self.reward, self.died

    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.snake.direction = random.choice(list(Direction)).value # random direction
        if self.score[-1] > self.best_score:
            self.best_score = self.score[-1]
            self.best_score_reached = True
        self.death += 1
        self.plot()
        self.mean_score.append(0)
        self.score.append(0)

    # function to get the current state of the snake agent
    def get_state(self):
        head_x, head_y = self.snake.body[0]
        if len(self.snake.body) > 1:
            behind_head = self.snake.body[1]
        food_x, food_y = self.food.position
        direction = self.snake.direction

        # check border danger one grid in the future in each direction relative to the head
        danger_right = (head_x + GRID_SIZE >= WIDTH)
        danger_left = (head_x - GRID_SIZE < 0)
        danger_up = (head_y - GRID_SIZE < 0)
        danger_down = (head_y + GRID_SIZE >= HEIGHT)

        # check self-collision danger one grid in the future in each direction relative to the head
        danger_right_self = ((head_x + GRID_SIZE, head_y) in self.snake.body) and (head_x + GRID_SIZE, head_y) != behind_head
        danger_left_self = (head_x - GRID_SIZE, head_y) in self.snake.body and (head_x - GRID_SIZE, head_y) != behind_head
        danger_up_self = (head_x, head_y - GRID_SIZE) in self.snake.body and (head_x, head_y - GRID_SIZE) != behind_head
        danger_down_self = (head_x, head_y + GRID_SIZE) in self.snake.body and (head_x, head_y + GRID_SIZE) != behind_head

        # position of the food relative to the head
        food_right = (food_x > head_x)
        food_left = (food_x < head_x)
        food_up = (food_y < head_y)
        food_down = (food_y > head_y)

        # current direction of the snake
        right = (direction == Direction.RIGHT.value)
        left = (direction == Direction.LEFT.value)
        up = (direction == Direction.UP.value)
        down = (direction == Direction.DOWN.value)

        # current lenght of the itself
        snake_length = len(self.snake.body) / (WIDTH * HEIGHT / GRID_SIZE**2)  # Normalize length

        # state
        state = [right, left, up, down,
                 food_right, food_left, food_up, food_down,
                 danger_right, danger_left, danger_up, danger_down,
                 danger_right_self, danger_left_self, danger_up_self, danger_down_self,
                 snake_length] # all values are 0 or 1
        return state
    
    # helper function to convert one hot coded action into valid movement
    def decode(self, action):
        directions = [Direction.UP.value, Direction.RIGHT.value, Direction.DOWN.value, Direction.LEFT.value] # [0, 1, 2, 3]: UP -> RIGHT -> DOWN -> LEFT (clock-wise)

        idx = directions.index(self.snake.direction) # get the index in the list 'directions' corresponding to the current snake direciton

        # action = [straigh, turn left, turn right] -> 0 = straight, 1 = turn left, 2 = turn right
        if action == 0:
            return self.snake.direction
        elif action == 1:
            return directions[(idx - 1) % 4] # counter clock-wise rotation
        elif action == 2:
            return directions[(idx + 1) % 4] # clock-wise rotation
    
    # plot the curves to monitor learning
    plt.ion()
    def plot(self):
        # calculate the mean score value
        total_score = sum(self.score)
        mean_score = total_score / min(MAX_BUFFER, self.death)
        self.mean_score[-1] = mean_score

        plt.figure("Score plot")  
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Number of deaths")
        plt.ylabel("Score")
        plt.plot(self.score, label="Score per death", color="blue", marker='o')
        plt.plot(self.mean_score, label="Mean score", color="orange", marker='x')
        plt.legend()
        plt.ylim(ymin=0)
        plt.text(len(self.score)-1, self.score[-1], str(self.score[-1])) # write the value of the last element
        plt.text(len(self.mean_score)-1, self.mean_score[-1], str(self.mean_score[-1])) # write the value of the last element
        plt.pause(0.1)

    def draw(self, screen, iterations=0):
        # Draw snake and food
        screen.fill(BLACK)
        self.snake.draw(screen)
        self.food.draw(screen)

        # Render texts and boxes
        self.ui.draw_text(screen, f"Score: {self.score[-1]}", (0, 0))
        self.ui.draw_text(screen, f"deaths: {self.death}", (0, 25))
        self.ui.draw_text(screen, f"Best score: {self.best_score}", (0, 50))
        self.ui.draw_text(screen, f"Time: {self.elapsed_time}s", (0, 75))
        self.ui.draw_text(screen, f"Iterations: {iterations}", (0, 100))
