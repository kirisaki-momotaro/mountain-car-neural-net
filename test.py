import gym
import torch
from itertools import count
from trainer import Trainer
env = gym.make("MountainCar-v0", render_mode="human")

trainer = Trainer(is_testing=True)

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
while True:
    action = trainer.select_action(state,env)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device="cpu")
    done = terminated or truncated

    if terminated:
        next_state = None
        break
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device="cpu").unsqueeze(0)

    # Move to the next state
    state = next_state



