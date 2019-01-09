
from torch import tensor
from model import PolicyModel

class Env_store:

    def __init__(self, dim, state_dim):
        self.actions = tensor(dim)
        self.rewards = tensor(dim)
        self.advantage = tensor(dim)
        self.states = tensor(state_dim)
        self.network_output = None
        self.dones = tensor(dim)

    def populate(self, states, rewards, dones, model_output):
        self.states = tensor(states)
        self.rewards = rewards
        self.not_dones = tensor(1 - dones).unsqueeze(-1)
        self.network_output = model_output
