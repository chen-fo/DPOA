import torch

from Net_type import PolicyNet1, PolicyNet2, ValueNet1, ValueNet2


device = torch.device('cpu')

class PPO:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma):
        self.actor = PolicyNet2(state_dim, action_dim).to(device)
        self.critic = ValueNet2(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor, lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic, lr=critic_lr)
        self.action_dim = action_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
    def take_action(self, state, forbidden_actions):
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.actor(state)
        if len(forbidden_actions) != 0 and len(forbidden_actions) != self.action_dim:
            mask = torch.ones_like(probs)
            mask[forbidden_actions] = 0
            probs = probs * mask
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()