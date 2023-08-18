import numpy as np
import random
import copy
from collections import namedtuple, deque

import matplotlib.pyplot as plt

from nets.ddpg import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# HyperParemeter
BUFFER_SIZE = 2000  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3       # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transform
transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])


def preprocess(img: torch.Tensor):
    """
        img : [batch, 3, height, width] || [3, height, width]
    """
    pil_image = TF.to_pil_image(img[0] if img.dim() == 4 else img)
    return transformer(pil_image).unsqueeze(0) if img.dim() == 4 else transformer(pil_image)


class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, action_size, save_dir='./', buffer_size=2000, batch_size=32):
        """Initialize an Agent object.
        
        Params
        ======
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.save_dir = save_dir

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(action_size).to(device)
        self.actor_target = Actor(action_size).to(device)
        self.actor_optimizer = optim.Adam(lr=0.001, params=self.actor_local.parameters())

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(action_size).to(device)
        self.critic_target = Critic(action_size).to(device)
        self.critic_optimizer = optim.Adam(lr=0.001, params=self.critic_local.parameters())

        # Noise process
        self.noise = OUNoise(action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
        self.counter = 0
        self.learn_frequence = 32
    
    def step(self, state, action, reward, next_state):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state)
        self.counter = (self.counter+1) % self.learn_frequence

        # Learn, if enough samples are available in memory
        if (len(self.memory) > self.batch_size)  :
            experiences = self.memory.sample()
            critic_loss, actor_loss = self.learn(experiences, GAMMA)
        else:
            critic_loss, actor_loss = 0.0, 0.0

        return critic_loss, actor_loss

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(preprocess(state).to(device)).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            action += noise
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)               

        return critic_loss.item(), actor_loss.item()      

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self):
        torch.save(self.actor_local.state_dict(), self.save_dir + '/actor.pth')
        torch.save(self.critic_local.state_dict(), self.save_dir + '/critic.pth')

    def load(self, directory):
        self.actor_local.load_state_dict(torch.load(directory + '/actor.pth'))
        self.critic_local.load_state_dict(torch.load(directory + '/critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.1, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
    
    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.stack([preprocess(e.state) for e in experiences], dim=0).to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.stack([preprocess(e.next_state) for e in experiences]).to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)