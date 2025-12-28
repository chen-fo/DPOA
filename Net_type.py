import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cpu")
class Net1(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 64)
        self.fc2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(64, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        action_prob = self.out(x)
        return action_prob


class Net2(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net2,self).__init__()

        self.fc1 = nn.Linear(num_states, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        action_prob = self.out(x)
        return action_prob

class Net3(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(num_states, 96)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(96, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        action_prob = self.out(x)
        return action_prob

class PolicyNet1(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(PolicyNet1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class PolicyNet2(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(PolicyNet2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class ValueNet1(nn.Module):
    def __init__(self,state_dim):
        super(ValueNet1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class ValueNet2(nn.Module):
    def __init__(self,state_dim):
        super(ValueNet2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class ACNet1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACNet1, self).__init__()
        # 策略网络 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # 价值网络 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


class ACNet2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACNet2, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


class CandidateModel(nn.Module):
    def __init__(self, id, input_shape, output_shape):
        super(CandidateModel, self).__init__()

        id %= 50
        if 0 <= id < 5:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id + 5)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id + 5), out_features=output_shape)
            )       
        elif 5 <= id < 10:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id), out_features=2**(id)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id), out_features=output_shape)
            )
        elif 10 <= id < 14:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 5)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 5), out_features=2**(id - 4)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 4), out_features=output_shape)
            )
        elif 14 <= id < 18:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 8)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 8), out_features=2**(id - 7)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 7), out_features=output_shape),
            )
        elif 18 <= id < 23:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 13)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 13), out_features=2**(id - 13)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 13), out_features=2**(id - 13)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 13), out_features=output_shape)
            )
        elif 23 <= id < 27:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 18)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 18), out_features=2**(id - 17)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 17), out_features=2**(id - 16)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 16), out_features=output_shape)
            )
        elif 27 <= id < 31:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 20)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 20), out_features=2**(id - 21)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 21), out_features=2**(id - 22)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 22), out_features=output_shape)
            )
        elif 31 <= id < 36:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 26)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 26), out_features=2**(id - 26)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 26), out_features=2**(id - 26)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 26), out_features=2**(id - 26)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 26), out_features=output_shape)
            )
        elif 36 <= id < 39:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 31)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 31), out_features=2**(id - 30)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 30), out_features=2**(id - 29)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 29), out_features=2**(id - 28)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 28), out_features=output_shape)
            )
        elif 39 <= id < 42:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 31)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 31), out_features=2**(id - 32)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 32), out_features=2**(id - 33)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 33), out_features=2**(id - 34)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 34), out_features=output_shape)
            )
        elif 42 <= id < 47:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**(id - 37)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 37), out_features=2**(id - 37)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 37), out_features=2**(id - 37)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 37), out_features=2**(id - 37)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 37), out_features=2**(id - 37)),
                nn.ReLU(),
                nn.Linear(in_features=2**(id - 37), out_features=output_shape)
            )
        elif id == 47:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**5),
                nn.ReLU(),
                nn.Linear(in_features=2**5, out_features=2**6),
                nn.ReLU(),
                nn.Linear(in_features=2**6, out_features=2**7),
                nn.ReLU(),
                nn.Linear(in_features=2**7, out_features=2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=2**9),
                nn.ReLU(),
                nn.Linear(in_features=2**9, out_features=output_shape)
            )
        elif id == 48:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**9),
                nn.ReLU(),
                nn.Linear(in_features=2**9, out_features=2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=2**7),
                nn.ReLU(),
                nn.Linear(in_features=2**7, out_features=2**6),
                nn.ReLU(),
                nn.Linear(in_features=2**6, out_features=2**5),
                nn.ReLU(),
                nn.Linear(in_features=2**5, out_features=output_shape)
            )
        elif id == 49:
            self.fc = nn.Sequential(
                nn.Linear(in_features=input_shape, out_features= 2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=2**8),
                nn.ReLU(),
                nn.Linear(in_features=2**8, out_features=output_shape)
            )

    def forward(self, x):
        return self.fc(x)
