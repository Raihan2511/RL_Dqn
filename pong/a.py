import gymnasium as gym
print(gym.envs.registry.keys())
# import gymnasium as gym
# print([env for env in gym.envs.registry.keys() if 'Pong' in env or 'ALE' in env])