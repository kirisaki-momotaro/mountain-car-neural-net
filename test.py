import gym
import torch
from model import NeuralNet

# initialize environment
env = gym.make("MountainCar-v0", render_mode="human")
# instantiate a neural network
player = NeuralNet()
# load trained weights
player.load_the_model(weights_filename="models/pnet.pt")
# reset to start state
state, info = env.reset()
# transform state in a form to enter the neural network
state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
# run loop
while True:
    # ask the nn what action to take given the state
    action = player(state).max(1).indices.view(1, 1)
    # apply action
    observation, reward, terminated, truncated, _ = env.step(action.item())
    # get next state
    state = torch.tensor(observation, dtype=torch.float32, device="cpu").unsqueeze(0)
