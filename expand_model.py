import random
import numpy as np
import os
from tqdm import tqdm

from DQN_model import DQN
from Mnt import Mnt

def sample_list(action_vote):

    total_weight = sum(action_vote)

    probabilities = [x / total_weight for x in action_vote]

    sampled_index = random.choices(range(len(action_vote)), weights=probabilities, k=1)[0]

    return sampled_index


multi_model_action_select_type = 'sample' #or greedy

max_frames = 12000

sub_num = 1
opp_num = 1
agent_num = sub_num + opp_num
model_j_num = 1
explore_rate = 0.3

sub_path = f'expand_model/{model_j_num}/'
if not os.path.exists(sub_path):
    os.makedirs(sub_path)

env = Mnt(sub_num=sub_num, opp_num=opp_num)
sub_obs_size = env.sub_obs_size
opp_obs_size = env.opp_obs_size
action_space = env.action_space

sub_model = DQN(0, 1, sub_obs_size, action_space, True, max_frames * explore_rate)
opp_model_set = []

for m in range(model_j_num):
    file_head = 'candidate_model/3/'
    opp_model_set.append(DQN(m, 5, opp_obs_size, action_space))
    opp_path = file_head + f'id_{m}.pkl' 
    opp_model_set[m].load_model(opp_path)

frame = 0
trial = 0
suc = 0
out = 0
on_mine = 0
conflict = 0
be_capture = 0
overdue = 0
capture = 0
total_reward = 0
with tqdm(total=max_frames) as pbar:
    while frame < max_frames:
        states = env.reset()
        done = False
        truncated = False
        actions = [0] * agent_num
        while not done and not truncated:
            last_live = [0] * agent_num
            for agt in range(agent_num):
                if env.check_live(agt):
                    last_live[agt] = 1
                    if agt < sub_num:
                        actions[agt] = (sub_model.choose_action(states[agt]))[0]
                    else:
                        action_vote = [0, 0, 0, 0, 0]
                        for m in range(model_j_num):
                            a = (opp_model_set[m].choose_action(states[agt]))[0]
                            action_vote[a] += 1

                        if multi_model_action_select_type == 'sample':
                            actions[agt] = sample_list(action_vote)
                        elif multi_model_action_select_type == 'greedy':
                            actions[agt] = np.argmax(action_vote)

            next_states, rewards, dones, done, truncated, info = env.step(actions)

            for agt in range(sub_num):
                if last_live[agt]:
                    sub_model.store_transition(states[agt], actions[agt], rewards[agt], next_states[agt], dones[agt])
                    sub_model.update()
                    total_reward += rewards[agt]
                    frame += 1
                    pbar.update(1)
            states = next_states
        trial += 1
        for agt in range(sub_num):
            if info is None:
                exit(0)
            if info[agt] == 1:
                suc += 1
            elif info[agt] == -1:
                out += 1
            elif info[agt] == -2:
                on_mine += 1
            elif info[agt] == -3:
                conflict += 1
            elif info[agt] == -4:
                be_capture += 1
            elif info[agt] == -5:
                overdue += 1
        for agt in range(sub_num, agent_num):
            if info is not None:
                capture += info[agt]
        denom = sub_num * trial
        pbar.set_postfix(
            {'trial': '%d' % trial , 'suc': '%.3f%%' % (suc / denom), 'out': '%.3f%%' % (out / denom),
             'mine': '%.3f%%' % (on_mine / denom), 'capture_rate': '%.3f%%' % (capture / denom),
             'overdue': '%.3f%%' % (overdue / denom), 'conflict': '%.3f%%' % (conflict / denom),
             'reward': total_reward})

sub_model.save_model(sub_path + 'sub_model.pkl')
    

    



