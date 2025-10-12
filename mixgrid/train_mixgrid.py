import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import minigrid # Important to import this to register the environments
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

import numpy as np
import random
from collections import deque, namedtuple

# --- Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
NUM_EPISODES = 2000 # MiniGrid can take a few more episodes to learn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Replay Memory (Same as before) ---
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Simple DQN Model (The CartPole Brain) ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Setup Environment ---
# We use a wrapper to get a fully observable, smaller grid view.
# This makes it easier for the simple DQN to learn.
env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human") # Try with rendering on! It's fast enough.
env = FullyObsWrapper(env) # This gives the agent a full view of the small grid
env = ImgObsWrapper(env) # This converts the grid numbers into a format the network can use

# Get number of actions and observations
n_actions = env.action_space.n
state, info = env.reset()
# The observation is an image, so we flatten it into a vector
n_observations = state.flatten().shape[0]

# Initialize networks, optimizer, and memory
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # Flatten the state and create a tensor for the network
            state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    # (This function is identical to the one in the first CartPole script)
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch = Experience(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# --- Training Loop ---
for i_episode in range(NUM_EPISODES):
    state, info = env.reset()
    state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    while True:
        action = select_action(state.cpu().numpy()) # select_action expects a numpy array
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            if reward > 0:
                print(f"Episode {i_episode+1}: SOLVED! Total Reward = {total_reward:.2f}")
            else:
                print(f"Episode {i_episode+1}: Failed. Total Reward = {total_reward:.2f}")
            break

print('Complete')
env.close()