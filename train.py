import gym
import torch
from itertools import count
from trainer import Trainer

# env = gym.make("MountainCar-v0", render_mode="human")
env = gym.make("MountainCar-v0")
trainer = Trainer()
observation, info = env.reset(seed=42)
num_episodes = 100
for i_episode in range(num_episodes):
    print(f"episode :{i_episode}")
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
    for t in count():
        action = trainer.select_action(state, env)
        observation, reward, terminated, truncated, _ = env.step(action.item())

        position, velocity = observation
        reward += abs(velocity)

        # print("this thing is stupid....")
        reward = torch.tensor([reward], device="cpu")
        done = terminated or truncated

        if terminated:
            next_state = None
            break
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device="cpu").unsqueeze(0)

        # Store the transition in memory
        trainer.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        trainer.optmz_model(t)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = trainer.target_net.state_dict()
        policy_net_state_dict = trainer.policy_net.state_dict()
        for key in policy_net_state_dict:  # TAU
            target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
        trainer.target_net.load_state_dict(target_net_state_dict)

print('Complete')
