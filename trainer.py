from model import NeuralNet
from memory import ReplayMemory, Transition
import torch.optim as optim
import torch
import random
import math
import torch.nn as nn


class Trainer:
    def __init__(self,
                 batch_size=128,
                 gamma=0.99,
                 epsilon_start=0.9,
                 epsilon_min=0.05,
                 epsilon_decay=1000,
                 LR=1e-4,
                 TAU=0.005,
                 number_of_actions=3,
                 number_of_observations=2):

        self.policy_net = NeuralNet().to("cpu")
        self.target_net = NeuralNet().to("cpu")
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.batch_size = batch_size,
        self.gamma = gamma,
        self.epsilon_start = epsilon_start,
        self.epsilon_min = epsilon_min,
        self.epsilon_decay = epsilon_decay,
        self.number_of_actions = number_of_actions,
        self.number_of_observations = number_of_observations
        self.steps_done = 0
        self.TAU = TAU

    # decide based on eps_threshold value between expected optimal and random action
    def select_action(self, state, env):
        sample = random.random()
        # decreases over time moving from exploration to exploitation
        eps_threshold = 0.05 + (0.9 - 0.05) * \
                        math.exp(-1. * self.steps_done / 1000)
        self.steps_done += 1
        # set minimum threshold
        # a small amount of randomness can be beneficial even for later runs
        if eps_threshold < 0.05:
            eps_threshold = 0.05
        # decide whether to explore or exploit
        if sample > eps_threshold:
            with torch.no_grad():
                # expected optimal action
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # random action
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

    # optimizes weights and biases of the network
    def optmz_model(self, turn_num):
        if len(self.memory) < 128: # if experiences number not enough don't optimize
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device="cpu", dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1})
        next_state_values = torch.zeros(self.batch_size, device="cpu")
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * 0.99) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        if turn_num % 1000 == 0:
            self.policy_net.save_the_model(weights_filename="models/pnet.pt")

