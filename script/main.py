import numpy as np
import random
from tqdm import tqdm

from DQN_model import DQN
from Mnt import Mnt
from no_sensor.belief_filter import BeliefFilter

max_trials = 2000
interval = 100

predict_mode = 0

#MTA
predict_num = 10
predict_model = []
predict_belief = []

#pomdp
prior_list = []


sub_num = 1
opp_num = 1
agent_num = sub_num + opp_num
agents = []

sub_path = ''
opp_path = ''

env = Mnt(sub_num=sub_num, opp_num=opp_num)
sub_obs_size = env.sub_obs_size
opp_obs_size = env.opp_obs_size
action_space = env.action_space

for i in range(agent_num):
    if i < sub_num:
        agents.append(DQN(i, 1, sub_obs_size, action_space, True))
    else:
        agents.append(DQN(i, 3, opp_obs_size, action_space, True))

if predict_mode == 1:
    head = 'candidate_model/3/'
    for m in range(predict_num):
        path = head + f'id_{m}.pkl'
        predict_model.append(DQN(m, 5, opp_obs_size, action_space))
        predict_model[m].load_model(path)

    predict_belief = [1] * predict_num

elif predict_mode == 2:

    for i in range(sub_num):
        i_pre = []
        priors = []
        for j in range(opp_num):
            i_pre.append(BeliefFilter(predict_num, action_space, 1))
            priors.append(i_pre[j].prior)
        predict_model.append(i_pre)
        prior_list.append(priors)


success_rate = []

for k in range(int(max_trials / interval)):
    trial = 0
    suc = 0
    out = 0
    on_mine = 0
    conflict = 0
    be_capture = 0
    overdue = 0
    capture = 0

    with tqdm(total=interval, desc='Iteration %d' % k) as pbar:
        total_reward = 0
        print(total_reward)
        while trial < interval:
            states = env.reset()
            done = False
            truncated = False
            actions = [0] * agent_num
            while not done and not truncated:
                last_live = [0] * agent_num
                for agt in range(sub_num, agent_num):
                    if env.check_live(agt):
                        last_live[agt] = 1
                        actions[agt] = agents[agt].choose_action(states[agt])[0]
                for agt in range(sub_num):
                    if env.check_live(agt):
                        last_live[agt] = 1
                        action_list = agents[agt].choose_action(states[agt])
                        if predict_mode:
                            agent_around = env.who_around(agt)
                            if predict_mode == 1:
                                for i in agent_around:
                                    predict_action_list = [0, 0, 0, 0, 0]
                                    for m in range(predict_num):
                                        idx = (predict_model[m].choose_action(states[i]))[0]
                                        predict_action_list[idx] += predict_belief[m]
                                        if idx == actions[i]:
                                            predict_belief[m] += 1
                                        else:
                                            predict_belief[m] -= 1

                                    predict_action = np.argmax(predict_action_list)
                                    action_list = [a for a in action_list if env.judge_move(agt, i, a, predict_action)]

                            elif predict_mode == 2:
                                for i in agent_around:
                                    predict_action, prior = predict_model[agt][i-sub_num].update([states[i]], prior_list[agt][i-sub_num])
                                    prior_list[agt][i] = prior
                                    action_list = [a for a in action_list if env.judge_move(agt, i, action_list[a], predict_action)]

                        if len(action_list) == 0:
                            actions[agt] = random.choice(range(5))
                        else:
                            actions[agt] = action_list[0]
                next_states, rewards, dones, done, truncated, info = env.step(actions)

                for agt in range(agent_num):
                    if last_live[agt]:
                        agents[agt].store_transition(states[agt], actions[agt], rewards[agt], next_states[agt], dones[agt])
                        agents[agt].update()
                        if agt < sub_num:
                            total_reward += rewards[agt]
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

            assert be_capture == capture
            assert suc + out + on_mine + conflict + be_capture + overdue == trial * sub_num
            denom = sub_num * interval

            pbar.set_postfix({'trial': '%d' % (trial + interval * k), 'suc': '%.2f%%' % (suc / denom), 'out': '%.2f%%' % (out / denom), 'mine': '%.2f%%' % (on_mine / denom), 'capture_rate': '%.2f%%' % (capture / denom), 'overdue': '%.2f%%' % (overdue / denom), 'conflict': '%.2f%%' % (conflict / denom), 'reward': total_reward})
            pbar.update(1)

    success_rate.append(round(suc / (interval * sub_num), 3))

print(success_rate)




