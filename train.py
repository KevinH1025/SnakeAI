import torch
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from collections import deque

MAX_BUFFER = 1000

class Trainer:
    def __init__(self, model, learning_rate, gamma):
        self.gamma = gamma # discount fasctor
        self.main_model = model # main network -> need to be optimized
        self.target_model = copy.deepcopy(model) # copy of the main network -> works as reference 
        self.criterion = nn.MSELoss() # MSE loss
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=learning_rate) # Adam optimizer
        self.min_lr = 1e-6
        self.lr_decay = 0.9
        self.decay_tolerance = 0

        # for the plot
        self.loss = deque(maxlen=MAX_BUFFER)
        self.average_loss = deque(maxlen=MAX_BUFFER)
        self.learning_rate = deque([learning_rate], maxlen=MAX_BUFFER)


    def train_QNet(self, batch, iterations):
        self.optimizer.zero_grad() # zero the grad
        states, actions, rewards, next_states, game_over = zip(*batch) # collects each attribute of every sample together
        
        # convert them into torch tensor
        states = torch.tensor(states, dtype=torch.float) # shape: (batch_size, state_size)
        actions = torch.tensor(actions, dtype=torch.long) # shape: (batch_size,)
        rewards = torch.tensor(rewards, dtype=torch.float) # shape: (batch_size,)
        next_states = torch.tensor(next_states, dtype=torch.float) # shape: (batch_size, state_size)
        game_over = torch.tensor(game_over, dtype=torch.float) # shape: (batch_size,)

        # get Q values for the current state
        output = self.main_model.forward(states) # shape: (batch_size, 3)

        # select Q values for the corresponding action
        actions = actions.unsqueeze(1) # shape: (batch_size, 1)
        Q_value = output.gather(1, actions).squeeze(1) # shape: (batch_size, 1) -> squeeze -> (batch_size,)

        # compute target Q value (target neural network). It does not need gradients, since this network is only for reference
        with torch.no_grad():
            target_output = self.target_model(states) # shape: (batch_size, 3)
            max_next_Q_values = torch.max(target_output, dim=1)[0] # max returns both values and indices, we need only values, hence [0] -> shape: (batch_size,)
            target_Q_values = rewards + self.gamma * max_next_Q_values * (1 - game_over) # perform element-wise mul.

        # compute loss and the gradient update
        loss = self.criterion(target_Q_values, Q_value) # loss
        # used for plotting
        self.loss.append(loss.item())
        #loss = torch.clamp(loss, -1, 1)  # Prevent extreme values
        loss.backward() # backpropagation
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.0)

        # calc average loss
        self.average_loss.append(sum(self.loss)/iterations)

        # update step    
        self.optimizer.step() 

        self.scheduler()

    # Learning rate decay
    def scheduler(self):
        # after x updates decay the rate
        self.decay_tolerance += 1
        if self.decay_tolerance == 10_000:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(self.min_lr, param_group['lr'] * self.lr_decay)
            self.decay_tolerance = 0

        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

    # plot the loss and learning rate
    plt.ion()
    def plot(self):
        # loss
        plt.figure("Loss plot")  
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.plot(range(1, len(self.loss) + 1)[::10], list(self.loss)[::10], label="Current loss", color="blue", marker='o')
        plt.plot(range(1, len(self.average_loss) + 1)[::10], list(self.average_loss)[::10], label="Average loss", color="orange", marker='x')
        plt.legend()
        plt.ylim(ymin=0)
        plt.text(len(self.loss)-1, self.loss[-1], str(self.loss[-1])) # write the value of the last element
        plt.text(len(self.average_loss)-1, self.average_loss[-1], str(self.average_loss[-1])) # write the value of the last element
        # learning rate
        plt.figure("Learning rate plot")  
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Number of iterations")
        plt.ylabel("Learning rate")
        plt.plot(range(1, len(self.learning_rate) + 1)[::10], list(self.learning_rate)[::10], label="Learning rate", color="Blue", marker='x')
        plt.legend()
        plt.ylim(ymin=0)
        plt.text(len(self.learning_rate)-1, self.learning_rate[-1], str(self.learning_rate[-1])) # write the value of the last element

        plt.pause(0.001)
        plt.draw()

    # target network copies the weights of main network
    def update_targetNet(self):
        self.target_model.load_state_dict(self.main_model.state_dict())

    # save model after every best score
    def save_models(self, best_score_reached):
        if best_score_reached:
            print("best score reached!")
            torch.save(self.main_model.state_dict(), 'model/main_model.pth')
            torch.save(self.target_model.state_dict(), 'model/target_model.pth')
            return True
        return False