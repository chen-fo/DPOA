import numpy as np
import random

from DQN_model import DQN
from no_sensor.Maze import Maze
from no_sensor.belief_filter import BeliefFilter

max_steps = 30
max_trials = 2000
interval = 100
num_state_sub = 20
num_state_opp = 13
num_actions = 5

predict_mode = 0
prior_list = []

predict_num = 10
predict_model = []
predict_belief = []

agent_num = 2
sub_num = 1
opp_num = 1
agent_type = []
agents = []
sub_path = 'single_models/sub_model.pkl'
opp_path = 'single_models/oppparameter1.pkl'

for i in range(agent_num):
    if i < sub_num:
        agent_type.append(1)
        agents.append(DQN(i, 1, num_state_sub, num_actions))
        agents[i].loadModel(sub_path)
    else:
        agent_type.append(2)
        agents.append(DQN(i, 3, num_state_opp, num_actions))
        agents[i].loadModel(opp_path)
if predict_mode == 1:
    head = 'candidate_model/3/'
    for m in range(predict_num):
        
        path = head + f'id_{m}.pkl'
        predict_model.append(DQN(m, 5, num_state_opp, num_actions))
        predict_model[m].loadModel(path)

    predict_belief = [1] * predict_num

elif predict_mode == 2:

    for i in range(sub_num):
        i_pre = []
        priors = []
        for j in range(opp_num):
            i_pre.append(BeliefFilter(predict_num, num_actions, 1))
            priors.append(i_pre[j].prior)
        predict_model.append(i_pre)
        prior_list.append(priors)

maze = Maze(agent_num, agent_type)

trial = 0
suc = 0
success_rate = []


while trial < max_trials:
    step = 0
    maze.refreshMaze(agent_num, agent_type)
    states = maze.get_all_State(agent_type)
    actions = [0] * agent_num
    rewards = [0] * agent_num
    while step < max_steps and not maze.endState_all(agent_type):
        
        for agt in range(sub_num, agent_num): 
            if not maze.checkendState(agt):
                actions[agt] = agents[agt].choose_action(states[agt])[0]
        for agt in range(sub_num):
            if not maze.checkendState(agt):
                action_list = agents[agt].choose_action(states[agt])
                if predict_mode:
                    agent_around = maze.who_around(agt, agent_type)
                    if predict_mode == 1:
                        for id in agent_around:
                            predict_action_list = [0, 0, 0, 0, 0]
                            for m in range(predict_num):
                                idx = (predict_model[m].choose_action(states[id]))[0]
                                predict_action_list[idx] += predict_belief[m]
                                if idx == actions[id]:
                                    predict_belief[m] += 1
                                else:
                                    predict_belief[m] -= 1

                            predict_action = np.argmax(predict_action_list)
                            for a in action_list:
                                if not maze.JudgeMove(agt, id, action_list[a], predict_action - 2):
                                    action_list.pop(a)

                    elif predict_mode == 2:
                        for id in agent_around:
                            opp_id = id - sub_num
                            predict_action, prior = predict_model[agt][opp_id].update(states[id], prior_list[agt][opp_id])
                            prior_list[agt][opp_id] = prior
                            for a in action_list:
                                if not maze.JudgeMove(agt, id, action_list[a], predict_action - 2):
                                    action_list.pop(a)

                if len(action_list) == 0:
                    actions[agt] = random.choice(range(5))
                else:
                    actions[agt] = action_list[0]

        for agt in range(agent_num):
            if not maze.checkendState(agt):
                maze.move(agt, actions[agt] - 2)

        next_states = maze.get_all_State(agent_type)

        for agt in range(sub_num):
            if not maze.checkendState(agt):
                reward = maze.get_i_Reward(agt, agent_type)
                rewards[agt] = reward
                agents[agt].store_transition(states[agt], actions[agt], reward, next_states[agt])
            if agents[agt].memory_counter >= DQN.batch_size and agent_type[agt] == 1:
                #agents[agt].learn()
                a=0

        step += 1
        states = next_states
    trial += 1
    for agt in range(sub_num):
        if rewards[agt] == 10:
            suc += 1
    if trial % interval == 0:
        success_rate.append(round(suc / (interval * sub_num), 2))
        print(f'trial:{trial}, success rate:{suc / (interval * sub_num)}')
        suc = 0
print(success_rate)
    

