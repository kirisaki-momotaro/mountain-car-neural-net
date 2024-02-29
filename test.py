import gym
import torch
from model import NeuralNet
from itertools import count
from trainer import Trainer

env = gym.make("MountainCar-v0", render_mode="human")
player = NeuralNet()
player.load_the_model(weights_filename="models/pnet.pt")
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
while True:
    #print(state)
    action = player(state).max(1).indices.view(1, 1)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    state = torch.tensor(observation, dtype=torch.float32, device="cpu").unsqueeze(0)
    #print(state)
