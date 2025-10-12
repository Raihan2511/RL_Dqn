import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple
import cv2 # OpenCV for image processing

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. Preprocessing Wrapper for Atari Environment ---
# This is a crucial new part for handling pixel data.
class AtariWrapper(gym.Wrapper):
    """
    Wrapper for Atari environments to preprocess frames:
    1. Grayscale: Convert RGB frames to grayscale.
    2. Resize: Downsample frames to 84x84.
    3. Stack Frames: Stack 4 consecutive frames to give the agent a sense of motion.
    """
    def __init__(self, env, k=4):
        super(AtariWrapper, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        # Override the observation space
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=((k,) + (84, 84)), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(self._preprocess(obs))
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._preprocess(obs))
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        # Stack frames along a new dimension and move to PyTorch tensor
        return np.stack(self.frames, axis=0)

    def _preprocess(self, frame):
        # Convert to grayscale and resize
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        return resized_frame


# --- 2. Replay Memory (Same as before) ---
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- 3. The CNN Brain ---
class DQN_CNN(nn.Module):
    def __init__(self, n_actions):
        super(DQN_CNN, self).__init__()
        # Input is a stack of 4 grayscale 84x84 frames
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the flattened size after conv layers
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        linear_input_size = 64 * 7 * 7 # Based on 84x84 input
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # The input x is (batch, 4, 84, 84). Normalize it.
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten the output
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- 4. Main Training Logic ---

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 1000 # How often to update the target network (in steps)
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-4
NUM_EPISODES = 10000

# Setup environment
# env = gym.make("Pong-v5")
env = gym.make("ALE/Pong-v5")
env = AtariWrapper(env)

n_actions = env.action_space.n
initial_screen = env.reset()[0]
_, screen_channels, screen_height, screen_width = initial_screen.shape

# Initialize networks
policy_net = DQN_CNN(n_actions).to(device)
target_net = DQN_CNN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # Target network is only for evaluation

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # state needs to be (1, 4, 84, 84)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            return policy_net(state_tensor).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Experience(*zip(*transitions))

    # Convert to PyTorch tensors
    state_batch = torch.from_numpy(np.array(batch.state)).float().to(device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # For next states, we want to check if they are final
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = None
    if non_final_mask.sum() > 0:
        non_final_next_states = torch.from_numpy(np.array([s for s in batch.next_state if s is not None])).float().to(device)

    # Compute V(s_{t+1}) for all non-final next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states is not None:
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1) # Gradient clipping
    optimizer.step()

# --- Training Loop ---
for i_episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    while True:
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward
        done = terminated or truncated

        # Convert reward to a tensor
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)

        if done:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward_tensor, done)

        state = next_state

        # Perform one step of the optimization
        optimize_model()

        if done:
            print(f"Episode {i_episode+1}, Steps: {steps_done}, Reward: {episode_reward}, Epsilon: {EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY):.4f}")
            break

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training complete.")
env.close()