import os
from tqdm import tqdm

from  Mnt import Mnt
from DQN_model import DQN


class TrueOppModel:
    agent_model = []
    sub_net = 'DQN'
    opp_net = 'DQN'

    max_frames = 8000

    def __init__(self, sub_num, opp_num, itera_num):
        self.agent_num = sub_num + opp_num
        self.sub_num = sub_num
        self.opp_num = opp_num
        self.env = Mnt(sub_num=sub_num, opp_num=opp_num)
        self.sub_obs_size = self.env.sub_obs_size
        self.opp_obs_size = self.env.opp_obs_size
        self.action_space = self.env.action_space

        self.itera_num = itera_num


        dir_path = f'single_models/{itera_num}/'
        if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print("have created directory", dir_path)

        self.sub_save_path = f'single_models/{itera_num}/sub_model_term.pkl'
        self.opp_save_path = f'single_models/{itera_num}/opp_model_{self.max_frames}.pkl'

        if self.sub_net == 'DQN':
            self.agent_model.append(DQN(0, 1, self.sub_obs_size, self.action_space, True, self.max_frames / 2))
        else:
            pass

        if self.opp_net == 'DQN':
            self.agent_model.append(DQN(0, 3, self.opp_obs_size, self.action_space, True, self.max_frames / 2))
        else:
            pass

        for i in range(itera_num):
            if i != 0:
                self.agent_model[1].load_model(self.opp_save_path)

            self.agent_model[0].buffer.buffer.clear()
            self.train(1)
            self.agent_model[0].load_model(self.sub_save_path)
            self.agent_model[1].buffer.buffer.clear()
            self.train(2)

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
        total_reward = 0
        frame = 0
        with tqdm(total=self.max_frames) as pbar:

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
                    pbar.set_postfix({'trial' : '%d' % trial, 'suc': '%.2f%%' % (suc/self.sub_num/trial), 'reward': total_reward})
                else:
                    pbar.set_postfix(
                        {'trial': '%d' % trial, 'captrue': '%.2f%%' % (suc / self.sub_num / trial), 'reward': total_reward})
            self.agent_model[train_type - 1].save_model(save_path)
if __name__ == "__main__":
    TrueOppModel(1, 1, 3)


                        

    
        

