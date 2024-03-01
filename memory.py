from collections import namedtuple, deque
import random

# a way to promptly create a class having the transition attributes
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# the memory where experiences are stored
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    #add a new transition
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    # sample a batch of transitions to use in training
    def sample(self, batch_size):
        return random.sample(self.memory, 128)

    def __len__(self):
        return len(self.memory)