from experience_buffer import MemoryBuffer
from model import DQN
from train import Trainer
import random
import torch
import os

MAX_BUFFER = 200_000

class SnakeAgent:
    def __init__(self, state_size=None, epsilon=1, randomess_decay=0.98, min_randomness=0.00001, discount=0.9, learning_rate=0.001):
        self.epsilon_rate = randomess_decay # exploration decay 
        self.epsilon = epsilon # ramdomness to start with
        self.epsilon_update = 30 # update the epsilon after x steps 
        self.epsilon_min = min_randomness # min. randomness 
        self.epsilon_count = 0 # counter for epsilon

        self.buffer = MemoryBuffer(MAX_BUFFER)
        self.gamma = discount # discount factor for Q function
        self.model = DQN(state_size) # model defined in model.py
        self.trainer = Trainer(self.model, learning_rate, self.gamma) # defines training for the model
        self.main_nn_update = 4 # every x steps we train and update the model
        self.main_nn_count = 0 # counter for steps
        self.target_nn_update = 1000 # every x steps we train and update the target model
        self.target_nn_count = 0 # counter for steps
        self.total_iterations = 0 # total training steps

    # get action for the agent after each step (frame)
    def get_action(self, state):
        # action is done randomly for exploration. Decay over time depending on the number of elements in the buffer
        if random.random() < self.epsilon:
            idx = random.randint(0, 2)
            return idx # direction = [0, 0, 0] -> [straight, turn right, turn left]
        
        # get action from the neural network (exploitation) 
        else:
            self.model.eval() # switch model to eval. mode for the running average of mean and variance of batch norm
            with torch.no_grad():
                idx = torch.argmax(self.model.forward(state)).item() # get the index of the higest Q value and use item() to convert it to standard integer 
                return idx # direction = [0, 0, 0] -> [straight, turn right, turn left]
    
    # train the agent on the past experience after certain steps
    def training(self):
        self.main_nn_count += 1

        # update the main network
        if self.main_nn_count == self.main_nn_update:
            self.target_nn_count += 1
            self.total_iterations += 1
            batch = self.buffer.get_batch() # get the batch
            self.trainer.train_QNet(batch, self.total_iterations) # execute one training step
            self.main_nn_count = 0
            #if self.total_iterations % 10 == 0:
            #    self.trainer.plot() # plot the loss
        
        # update the target network
        if self.target_nn_count == self.target_nn_update:
            self.trainer.update_targetNet()
            self.target_nn_count = 0

        return self.total_iterations
    
    # saving each step in memory buffer for training
    def remember(self, state, action, reward, next_state, game_over):
        self.buffer.add(state, action, reward, next_state, game_over)
        self.epsilon_count += 1
        #print(len(self.buffer.buffer))
        
        # calling helper funciton for epsilon
        if self.epsilon_count == self.epsilon_update:
            self.update_epsilon()
            self.epsilon_count = 0
    
    # helper function to update the epsilon depending on the steps
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_rate)
        #print(self.epsilon)

    # load an existing model
    def load_model(self, main_model_path, target_model_path):
        if os.path.exists(main_model_path) and os.path.exists(target_model_path):
            self.model.load_state_dict(torch.load(main_model_path, weights_only=True))
            self.trainer.target_model.load_state_dict(torch.load(target_model_path, weights_only=True))
            print("Models loaded succsefully")
        else:
            print("Model file not found. Training a new model...")