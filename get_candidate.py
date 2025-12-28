import os
import argparse
from tqdm import tqdm

from DQN_model import DQN
from Mnt import Mnt


class CandidateModel:
    agent_model = []
    agent_type = []
    sub_net = 'DQN'
    opp_net = 'DQN'

    max_frames = 10000
    outset = 0

    def __init__(self, sub_num, opp_num, itera_num, num_candidate, outset):
        self.sub_num = sub_num
        self.opp_num = opp_num
        self.agent_num = sub_num + opp_num
        self.env = Mnt(sub_num=sub_num, opp_num=opp_num)
        self.sub_obs_size = self.env.sub_obs_size
        self.opp_obs_size = self.env.opp_obs_size
        self.action_space = self.env.action_space

        self.num_candidate = num_candidate
        self.outset = outset

        dir_path = f'candidate_model/{itera_num}/'
        if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        self.sub_save_path = f'candidate_model/sub_model.pkl'
        
        if self.sub_net == 'DQN':
            self.agent_model.append(DQN(0, 1, self.sub_obs_size, self.action_space, True, self.max_frames / 2))

        for m in range(self.num_candidate):
            self.model_id = m
            self.agent_model.append(DQN(m + outset, 5, self.opp_obs_size, self.action_space, True, self.max_frames / 2))
            self.opp_save_path = dir_path + f'id_{m + outset}.pkl'
            for i in range(itera_num):
                if i != 0:
                    self.agent_model[1].load_model(self.opp_save_path)

                self.agent_model[0].buffer.buffer.clear()
                self.train(1)
                self.agent_model[0].load_model(self.sub_save_path)
                self.agent_model[1].buffer.buffer.clear()
                self.train(2)
            self.agent_model.pop()


    def train(self, train_type):
        if train_type == 1:
            save_path = self.sub_save_path
            a = 0
            b = self.sub_num
        else:
            save_path = self.opp_save_path
            a = self.sub_num
            b = self.agent_num
        trial = 0
        suc = 0
        with tqdm(total=self.max_frames) as pbar:
            total_reward = 0
            frame = 0
            while frame < self.max_frames:
                states = self.env.reset()
                done = False
                truncated = False
                actions = [0] * self.agent_num
                while not done and not truncated:
                    last_live = [0] * self.agent_num
                    for agt in range(self.agent_num):
                        if self.env.check_live(agt):
                            last_live[agt] = 1
                            if agt < self.sub_num:
                                actions[agt] = (self.agent_model[0].choose_action(states[agt]))[0]
                            else:
                                actions[agt] = (self.agent_model[1].choose_action(states[agt]))[0]
                    next_states, rewards, dones, done, truncated, info = self.env.step(actions)

                    for agt in range(a, b):
                        if last_live[agt]:
                            self.agent_model[train_type - 1].store_transition(states[agt], actions[agt], rewards[agt], next_states[agt], dones[agt])
                            self.agent_model[train_type - 1].update()
                            total_reward += rewards[agt]
                            frame += 1
                            pbar.update(1)
                    states = next_states
                trial += 1
                for i in range(a, b):
                    if info[i] > 0:
                        suc += info[i]
                if train_type == 1:
                    pbar.set_postfix({'model_id' : self.model_id,'trial' : '%d' % trial, 'suc': '%.4f%%' % (suc/self.sub_num/trial), 'reward': total_reward})
                else:
                    pbar.set_postfix({'trial': '%d' % trial, 'captrue': '%.4f%%' % (suc / self.sub_num / trial), 'reward': total_reward})
            self.agent_model[train_type - 1].save_model(save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_candidate", default=1, type=int)
    parser.add_argument("--sub_num", default=1, type=int)
    parser.add_argument("--opp_num", default=1, type=int)
    parser.add_argument("--outset", default=0, type=int)
    parser.add_argument("--itera_num", default=3, type=int)
    args = parser.parse_args()
    CandidateModel(args.sub_num, args.opp_num, args.itera_num, args.num_candidate, args.outset)


                        


    
        

