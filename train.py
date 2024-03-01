import gym
import torch
from itertools import count
from trainer import Trainer

# env = gym.make("MountainCar-v0", render_mode="human")
# load environment
env = gym.make("MountainCar-v0")
# instantiate the agent responsible for the training
trainer = Trainer()
# reset environment
observation, info = env.reset(seed=42)
# train for num_episodes runs
num_episodes = 100
for i_episode in range(num_episodes):
    print(f"episode :{i_episode}")
    # Initialize the environment and get its start state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
    for t in count():
        # ask trainer what action to take (trainer decides based on epsilon value between expected optimal and random action)
        action = trainer.select_action(state, env)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        # give higher rewards for higher speeds
        position, velocity = observation
        reward += abs(velocity)
        # reward to device
        reward = torch.tensor([reward], device="cpu")
        done = terminated or truncated
        # if the car gets to the flag start next run
        if terminated:
            next_state = None
            break
        else:
            # set returned state as next state
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
        for key in policy_net_state_dict:  # compute Q-values
            target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
        trainer.target_net.load_state_dict(target_net_state_dict)

print('Complete')
