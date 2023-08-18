import numpy as np
import random
import copy
from collections import namedtuple, deque

import matplotlib.pyplot as plt

from nets.ddpg import Actor, Critic

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class TD3(object):
	def __init__(
		self,
		action_dim,
		max_action=1.0,
		discount=0.99,
		tau=0.005,
		policy_noise=0.1,
		noise_clip=0.5,
		policy_freq=2,
		buffer_size=2000,
		batch_size=32,
		save_dir='./'
	):

		self.actor = Actor(action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.actor_optimizer_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=20000, gamma=0.8)

		self.critic = Critic(action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.critic_optimizer_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=20000, gamma=0.8)

		self.memory = ReplayBuffer(action_dim, buffer_size, batch_size)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		self.save_dir = save_dir

	def add(self, state, action, reward, next_state):
		self.memory.add(state, action, reward, next_state)


	def select_action(self, state):
		with torch.no_grad():
			action = self.actor(preprocess(state).to(device)).cpu().data.numpy()	
		return action


	def train(self):
		self.total_it += 1

		# Sample replay buffer 
		state, action, reward, next_state = self.memory.sample()

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		self.critic_optimizer_scheduler.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			self.actor_optimizer_scheduler.step()


			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self):
		torch.save(self.critic.state_dict(), self.save_dir + "/critic")
		torch.save(self.critic_optimizer.state_dict(), self.save_dir + "/critic_optimizer")
		
		torch.save(self.actor.state_dict(), self.save_dir + "/actor")
		torch.save(self.actor_optimizer.state_dict(), self.save_dir + "/actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "/critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "/critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "/actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "/actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
