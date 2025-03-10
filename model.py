import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, output_size=3):
        super().__init__()

        # layer sizes
        self.input = input_size
        self.output = output_size

        self.dqn = nn.Sequential(
            
            nn.Linear(self.input, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, output_size)
        )

    def forward(self, x):
        if isinstance(x, list): # covert list to tensor
            # unsqueeze(0) adds extra dimention at pos 0 (batch). Important for Pytorch!  
            # float() convert the input. Pytorch expects floating-point
            x = torch.tensor(x, dtype=torch.float).unsqueeze(0)

        return self.dqn(x)
