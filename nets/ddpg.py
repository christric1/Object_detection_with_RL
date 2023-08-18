import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()

        self.resnet18 = torch.nn.Sequential(*(list(models.resnet18().children())[:-2]))

        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 512),
            nn.ReLU(),
            
            nn.Linear(512, action_size),
        )

        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):                                   
        """Build an actor (policy) network that maps states -> actions."""
        x = self.resnet18(state)
        x = x.view(-1, 512 * 7 * 7)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        x = self.fc(x)                                                                                                                                                                                                                                                                              
            
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""                                     

    def __init__(self, action_size):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()
        
        self.resnet18 = torch.nn.Sequential(*(list(models.resnet18().children())[:-2]))

        self.fc1 = nn.Sequential(
            nn.Linear(7*7*512 + action_size, 512),
            nn.ReLU(),

            nn.Linear(512, 1),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(7*7*512 + action_size, 512),
            nn.ReLU(),

            nn.Linear(512, 1),
            nn.ReLU(),
        )

        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.resnet18(state)
        x = x.view(-1, 512 * 7 * 7)
        x = torch.cat([x, action], dim=1)

        q1 = self.fc1(x)
        q2 = self.fc2(x)

        return q1, q2

    def Q1(self, state, action):
        x = self.resnet18(state)
        x = x.view(-1, 512 * 7 * 7)
        x = torch.cat([x, action], dim=1)

        q1 = self.fc1(x) 

        return q1                   
