# play.py - The script to use your trained agent

import torch
import gymnasium as gym
import time
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
# from RL_DQN.cartpole.train_cartpole import DQN 
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

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode="human")

# Get size of state and action space 
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

# Create the network and load the saved weights
policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load('cartpole_model.pth'))
policy_net.eval() # Set the network to evaluation mode (very important!)

# --- PLAYING LOOP ---
print("Watching the trained agent play...")
# while True:
#     with torch.no_grad(): # We don't need to calculate gradients
#         # Convert state to tensor and add a batch dimension
#         state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
#         # Get the best action from the network
#         action = policy_net(state_tensor).max(1)[1].view(1, 1)

#     # Perform the action in the environment
#     observation, reward, terminated, truncated, _ = env.step(action.item())
#     state = observation
#     done = terminated or truncated

#     if done:
#         print("Episode finished. Resetting.")
#         state, info = env.reset()
#         time.sleep(1) 
try:
    while True:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy_net(state_tensor).max(1)[1].view(1, 1)

        observation, reward, terminated, truncated, _ = env.step(action.item())
        state = observation
        done = terminated or truncated

        if done:
            print("Resetting.")
            state, info = env.reset()
            time.sleep(1.5)

except KeyboardInterrupt:
    print("Playback stopped by user.")

finally:
    # This ensures the environment window is closed cleanly
    env.close()