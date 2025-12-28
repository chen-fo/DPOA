import torch

from Net_type import Net3

device = torch.device('cpu')

batch_size = 64
gamma = 1


class DDQN:
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
):
        self.action_dim = action_dim
        self.q_net = Net3(state_dim, self.action_dim).to(device)
        self.target_q_net = Net3(state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-2)
        self.epsilon = epsilon
        self.target_update = 100
        self.count = 0


