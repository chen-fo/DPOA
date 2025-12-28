import numpy as np
import time
import random
import copy

from Mnt import Mnt
from DQN_model import DQN

random.seed(42)

def ve_simulate(sub_num, opp_num, models):
    eta = 0

    q, w, e = 10, 6, 4

    start_time = time.time()
    agent_num = sub_num + opp_num
    models_num = len(models)

    env = Mnt(sub_num=sub_num, opp_num=opp_num)
    sub_obs_size = env.sub_obs_size
    action_space = env.action_space

    env_num = 5
    env_paras = importance_sampling(env_num)
    scenario_num = env_num * models_num

    scenarios = 0
    values = np.zeros((models_num, scenario_num))
    sims = np.zeros((models_num, models_num))

    sub_agent = DQN(0, 1, sub_obs_size, action_space)
    for i in range(models_num):
        opp_agent = models[i]
        for ep in range(env_num):
            for m in range(models_num):
                states = env.reset(env_paras[ep])
                done = False
                truncated = False
                actions = [0] * agent_num
                while not done and not truncated:
                    for agt in range(sub_num):
                        if env.check_live(agt):
                            action_list = sub_agent.choose_action(states[agt], True)
                            agent_around = env.who_around(agt)
                            for other in agent_around:
                                predict_action = models[m].choose_action(states[other], True)[0]
                                action_list = [a for a in action_list if env.judge_move(agt, i, action_list[a], predict_action)]
                            if len(action_list) == 0:
                                actions[agt] = random.choice(range(action_space))
                            else:
                                actions[agt] = action_list[0]
                    for agt in range(sub_num, agent_num):
                        if env.check_live(agt):
                            actions[agt] = opp_agent.choose_action(states[agt])[0]
                    next_states, rewards, dones, done, truncated, info = env.step(actions)
                    states = next_states
                    for agt in range(sub_num):
                        values[m][scenarios] += rewards[agt]
            scenarios += 1


    for i in range(models_num):
        for j in range(i + 1, models_num):
            eq_num = 0

            for k in range(scenario_num):
                if abs(values[i][k] - values[j][k]) == eta:
                    eq_num += 1
            sims[i][j] = eq_num
            sims[j][i] = eq_num

    weight = [1] * models_num
    model_id = [i for i in range(models_num)]
    sim_test = copy.deepcopy(sims)

    while len(model_id) > q:

        two_max = np.unravel_index(np.argmax(sim_test), sim_test.shape)
        d = 0
        if values[model_id[two_max[0]]].sum() >= values[model_id[two_max[1]]].sum():
            d = 1
        weight[model_id[two_max[(d + 1) % 2]]] += weight[model_id[two_max[d]]]
        weight[model_id[two_max[d]]] = 0
        sim_test = np.delete(sim_test, two_max[d], axis=0)
        sim_test = np.delete(sim_test, two_max[d], axis=1)
        model_id.remove(model_id[two_max[d]])

    print(f'{q} scenrio model', model_id)
    print(f'{q} scenrio weight', weight)

    weight = [1] * models_num
    model_id = [i for i in range(models_num)]
    sim_test = copy.deepcopy(sims)
    while len(model_id) > w:
        two_max = np.unravel_index(np.argmax(sim_test), sim_test.shape)
        d = 0
        if values[model_id[two_max[0]]].sum() >= values[model_id[two_max[1]]].sum():
            d = 1
        weight[model_id[two_max[(d + 1) % 2]]] += weight[model_id[two_max[d]]]
        weight[model_id[two_max[d]]] = 0
        sim_test = np.delete(sim_test, two_max[d], axis=0)
        sim_test = np.delete(sim_test, two_max[d], axis=1)
        model_id.remove(model_id[two_max[d]])

    print(f'{w} scenrio model', model_id)
    print(f'{w} scenrio weight', weight)

    weight = [1] * models_num
    model_id = [i for i in range(models_num)]
    sim_test = copy.deepcopy(sims)
    while len(model_id) > e:
        two_max = np.unravel_index(np.argmax(sim_test), sim_test.shape)
        d = 0
        if values[model_id[two_max[0]]].sum() >= values[model_id[two_max[1]]].sum():
            d = 1
        weight[model_id[two_max[(d + 1) % 2]]] += weight[model_id[two_max[d]]]
        weight[model_id[two_max[d]]] = 0
        sim_test = np.delete(sim_test, two_max[d], axis=0)
        sim_test = np.delete(sim_test, two_max[d], axis=1)
        model_id.remove(model_id[two_max[d]])
    print(f'{e} scenrio model', model_id)
    print(f'{e} scenrio weight', weight)

    end_time = time.time()
    return end_time - start_time

def importance_sampling(num=5, mode=1):
    sub_num, opp_num = 1, 1
    agent_num = sub_num + opp_num
    env = Mnt(sub_num=sub_num, opp_num=opp_num)
    sub_obs_size = env.sub_obs_size
    action_space = env.action_space
    paras = []
    if mode == 1:
        for e in range(num):
            para = {'current': np.zeros((agent_num, 2)), 'bearing': [[0] * agent_num], 'target': [0, 0], 'mine': []}
            env.reset()
            for agt in range(agent_num):
                para['current'][agt][0] = env.current[agt][0]
                para['current'][agt][1] = env.current[agt][1]
                para['bearing'][agt] = env.bearing[agt]
            para['target'][0] = env.target[0]
            para['target'][1] = env.target[1]
            for i in range(env.size):
                for j in range(env.size):
                    if env.mines_map[i][j] == 1:
                        para['mine'].append([i, j])
            paras.append(para)
    elif mode == 2:
        reinforce_model = DQN(0, 1, sub_obs_size, action_space)
        for e in range(num * 100):
            para = {'current': np.zeros((agent_num, 2)), 'bearing': [[0] * agent_num], 'target': [0, 0], 'mine': []}
            env.reset()
            for agt in range(agent_num):
                para['current'][agt][0] = env.current[agt][0]
                para['current'][agt][1] = env.current[agt][1]
                para['bearing'][agt] = env.bearing[agt]
            para['target'][0] = env.target[0]
            para['target'][1] = env.target[1]
            for i in range(env.size):
                for j in range(env.size):
                    if env.mines_map[i][j] == 1:
                        para['mine'].append([i, j])
            paras.append(para)
        importance_list = []
        for para in paras:
            states = env.reset(para)
            discrepancy = 0
            for i in range(sub_num):
                discrepancy += reinforce_model.q_diff(states[i])
            importance_list.append(discrepancy)
        combined = list(zip(paras, importance_list))
        paras = [para for para, _ in sorted(combined, key=lambda x: x[1], reverse=True)[:num]]

    return paras
