import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Config:

    # Static Device declaration
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    def __init__(self):

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.action_size = 2
        self.state_dim = 8
        self.action_dim = 2
        #setting this to True disables training
        self.play_only = True

        self.lr = 1e-4
        self.discount = 0.97

        self.gae_tau = 0.95
        self.gradient_clip = 4.7
        self.rollout_length = 2000
        self.optimization_epochs = 10
        self.mini_batch_size = 200
        self.ppo_ratio_clip = 0.1
        self.log_interval = 100
        self.max_steps = 4e5

        self.entropy_weight = 0.09
        #self.logger = get_logger()
        self.num_workers = 2

        self.saved_checkpoint = 'checkpoint/ppo.pth'
