import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from collections import namedtuple

from torchsummary import summary

PolicyModel = namedtuple('PolicyModel',['log_prob', 'entropy', 'action','value'])
import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from collections import namedtuple

PolicyModel = namedtuple('PolicyModel',['log_prob', 'entropy', 'action','value'])

class PPONetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(PPONetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)


        #second_hidden_size = hidden_size - 100

        second_hidden_size = 500
        third = second_hidden_size - 100

        frames = 3 # each agent gets 3 frames
        agents = 2

        self.input_size = state_size * frames

        self.input = nn.Linear(self.input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, second_hidden_size)

        self.actor_body = nn.Linear(third, third)
        self.actor_head = nn.Linear(third, action_size)

        self.critic_body =  nn.Linear(third, third)
        self.critic_head = nn.Linear(third, 1)

        self.policy_body = nn.Linear(second_hidden_size, third)
        self.policy_head = nn.Linear(third, third)

        init_layers = [self.input, self.hidden, self.actor_body, self.critic_body, self.policy_body]
        self.init_weights(init_layers)

        self.batch_norm = nn.BatchNorm1d(second_hidden_size)
        self.batch_norm_input = nn.BatchNorm1d(hidden_size)

        self.alpha = nn.Linear(third, 2, bias=False)
        self.beta = nn.Linear(third, 2, bias=False)
        #
        # # init the networks....
        self.alpha.weight.data.mul_(0.125)
        # self.alpha.bias.data.mul_(0.1)
        # #
        self.beta.weight.data.mul_(0.125)
        # self.beta.bias.data.mul_(0.0)

        # self.alpha_param = nn.Parameter(torch.zeros(4))
        # self.alfa = nn.Parameter(torch.zeros(action_dim))

        self.std = nn.Parameter(torch.zeros(2))

        self.state_size = state_size

        device = 'cuda:0'
        self.to(device)

        summary(self, (1, self.input_size))

    def init_weights(self, layers):
        for layer in layers:
            nn.init.kaiming_normal_(layer.weight)
            layer.bias.data.mul_(0.1)


    def forward(self, state, action = None):
        x = state.view(-1, self.input_size)
        x = F.leaky_relu(self.batch_norm_input(self.input(x)))
        x = F.leaky_relu(self.batch_norm(self.hidden(x)))
        x = F.leaky_relu(self.policy_body(x))

        #act_x = F.leaky_relu(self.actor_body(x))
        act_x = F.tanh(self.actor_body(x))


        mean = F.tanh(self.actor_head(act_x))

        #alpha and beta parameters for Beta distribution
        alpha = F.softplus(self.alpha(act_x)) + 1
        beta = F.softplus(self.beta(act_x)) + 1


        # policy distribution - using Beta here which is a lot more efficient than Gaussian
        policy_dist = torch.distributions.Beta(alpha, beta)

        if action is None:
            action = policy_dist.sample()


        log_prob = policy_dist.log_prob(action).sum(-1).unsqueeze(-1)

        entropy = policy_dist.entropy().sum(-1).unsqueeze(-1)
        # entropy = (alpha - beta) / 2

        # critic value
        critic_x = F.leaky_relu(self.critic_body(x))
        value = self.critic_head(critic_x)

        #action = action.view(-1, 4)
        #print(action.shape)

        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': value}

        # return PolicyModel(log_prob, entropy, action, value)







