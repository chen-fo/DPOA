from Net_type import Net1, Net2, Net3, CandidateModel
import torch
import torch.nn as nn
import numpy as np
import random
import collections

device = torch.device("cpu")


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class DQN:
    batch_size = 64
    lr = 1e-3
    gamma = 0.95

    buffer_size = 5000
    minimal_size = 500
    target_update = 50
    count = 0

    epsilon_start = 1
    epsilon_time = 2000
    epsilon_end = 0.01


    def __init__(self, id, type, num_input, num_output, learn_flag = False, epsilon_time = 2000):
        self.id = id
        self.epsilon_time = epsilon_time
        self.set_learn_flag(learn_flag)

        self.num_state = num_input
        self.num_actions = num_output
        if type == 1:
            self.eval_net, self.target_net = Net1(num_input, num_output).to(device), Net1(num_input, num_output).to(device)
        elif type == 2:
            self.eval_net, self.target_net = Net2(num_input, num_output).to(device), Net2(num_input, num_output).to(device)
        elif type == 3:
            self.eval_net, self.target_net = Net3(num_input, num_output).to(device), Net3(num_input, num_output).to(device)
        elif type == 5:
            if id >= 50:
                self.lr = 0.0001 
            self.eval_net, self.target_net = CandidateModel(id, num_input, num_output).to(device), CandidateModel(id,num_input, num_output).to(device)

        else:
            print("type error")
            exit(0)

        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, deterministic=False):

        state = torch.FloatTensor(state).to(device)

        if self.epsilon > self.epsilon_end:
            self.epsilon -= (self.epsilon - self.epsilon_end) / self.epsilon_time
        if np.random.random() < self.epsilon and not deterministic:
            actions = [0, 1, 2, 3, 4]
            random.shuffle(actions)

        else:
            action_value = self.eval_net(state)
            _, index = action_value.sort(descending = True)
            actions = index.tolist()
        return actions

    def q_diff(self, state):
        state = torch.FloatTensor(state).to(device)
        action_value = self.eval_net(state)

        max_q_values, _ = torch.max(action_value, dim=1)
        min_q_values, _ = torch.min(action_value, dim=1)

        q_diff = (max_q_values - min_q_values).float()
        return q_diff

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):

        if self.buffer.size() > self.minimal_size:
            b_s, b_a, b_r, b_ns, b_d = self.buffer.sample()
            states = torch.tensor(b_s, dtype=torch.float).to(device)
            actions = torch.tensor(b_a).view(-1, 1).to(device)
            rewards = torch.tensor(b_r, dtype=torch.float).view(-1, 1).to(device)
            next_states = torch.tensor(b_ns, dtype=torch.float)
            dones = torch.tensor(b_d, dtype=torch.float).view(-1, 1).to(device)

            q_values = self.eval_net(states).gather(1, actions)
            max_actions = self.eval_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_net(next_states).gather(1, max_actions)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            loss = self.loss_func(q_values, q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.count % self.target_update ==0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.count += 1

    def save_model(self, outfile):
        torch.save(self.eval_net.state_dict(), outfile)

    def load_model(self, infile, trace=False):
        self.eval_net.load_state_dict(torch.load(infile, map_location=torch.device('cpu')))
        self.target_net.load_state_dict(torch.load(infile, map_location=torch.device('cpu')))
        if trace:
            print(f"have load model {infile}")

    def set_learn_flag(self, learn_flag):
        if learn_flag:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = self.epsilon_end




