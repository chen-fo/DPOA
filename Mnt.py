import random
import numpy as np

def merge_array(arr1, arr2):
    combine_list = []
    for row in arr1:
        combine_list.append(row)
    for row in arr2:
        combine_list.append(row)
    return combine_list

train_reward = True

class Mnt:
    def __init__(
            self,
            size=16,
            num_mines=8,
            sub_num=1,
            opp_num=1,
            sound=8,
            gamma=0.95,
            render=True
    ):

        self.size = size
        self.num_mines = num_mines
        self.sub_num = sub_num
        self.opp_num = opp_num
        self.agent_num = sub_num + opp_num
        self.sound = sound
        self.gamma = gamma
        self.render = render
        self.sub_obs_size = 1 + 2 + 5 + 5 + 1
        self.opp_obs_size = 1 + sub_num * 2 + 5 + 1
        self.action_space = 5

        self.steps = 0
        self.max_steps = 30

        self.map = None
        self.mines_map = None
        self.live = None
        self.info = None
        self.path = None

        self.current = np.zeros((self.agent_num, 2), dtype=int)
        self.pre_current = np.zeros((self.agent_num, 2), dtype=int)
        self.bearing = np.zeros(self.agent_num, dtype=int)
        self.target = np.zeros(2, dtype=int)


        #sub: 0 无事发生 1到达目的地 -1 出界 -2 踩雷 -3 碰撞 - 4 被捕 -5 过时
        #opp: 捕获量
        self.reset()

    def reset(self, para=None):
        self.steps = 0
        self.live = [True] * self.agent_num
        self.info = [0] * self.agent_num
        self.map = np.zeros((self.size, self.size), dtype=int)
        self.mines_map = np.zeros((self.size, self.size), dtype=int)
        self.path = np.zeros((self.agent_num, self.max_steps, 2))
        if para is None:
            for agt in range(self.agent_num):
                while True:
                    if agt >= self.sub_num:
                        x = random.randint(2, self.size - 3)
                        y = random.randint(2, self.size - 3)
                    else:
                        x = random.randint(0, self.size - 1)
                        if 2 <= x <= self.size - 3:
                            y = random.randint(0, 1) + random.randint(0, 1) * (self.size - 2)
                        else:
                            y = random.randint(0, self.size - 1)

                    self.current[agt][0] = x
                    self.current[agt][1] = y
                    self.pre_current[agt][0] = x
                    self.pre_current[agt][1] = y

                    if self.map[x][y] == 0:
                        self.map[x][y] = agt + 1
                        break

                self.path[agt][0][0] = self.current[agt][0]
                self.path[agt][0][1] = self.current[agt][1]

            while True:
                x = random.randint(2, self.size - 3)
                y = random.randint(2, self.size - 3)
                self.target[0] = x
                self.target[1] = y
                if self.map[x][y] == 0:
                    break

            for i in range(self.num_mines):
                while True:
                    x = random.randint(2, self.size - 3)
                    y = random.randint(2, self.size - 3)

                    if self.map[x][y] == 0 and self.mines_map[x][y] == 0 and x != self.target[0] or y != self.target[1]:
                        self.mines_map[x][y] = 1
                        break
            for agt in range(self.agent_num):
                self.bearing[agt] = random.randint(0, 7)
        else:
            for agt in range(self.agent_num):
                self.current[agt][0] = para['current'][agt][0]
                self.current[agt][1] = para['current'][agt][1]
                self.pre_current[agt][0] = self.current[agt][0]
                self.pre_current[agt][1] = self.current[agt][1]
                self.map[self.current[agt][0]][self.current[agt][1]] = agt + 1
                self.bearing[agt] = para['bearing'][agt]
            self.target[0] = para['target'][0]
            self.target[1] = para['target'][1]
            for m in range(len(para['mine'])):
                self.mines_map[para['mine'][m][0]][para['mine'][m][1]] = 1

        states = self.get_states()

        return states

    def step(self, actions):
        for i, action in enumerate(actions):
            if not self.live[i]:
                continue

            self.pre_current[i][0] = self.current[i][0]
            self.pre_current[i][1] = self.current[i][1]
            self.move(i, action)
        self.steps += 1
        rewards = self.update()
        dones = [not a for a in self.live]
        done = self.check_done()
        truncated = self.steps >= self.max_steps
        if done or truncated:
            info = self.get_info()
        else:
            info = None

        return self.get_states(), rewards, dones, done, truncated, info

    def move(self, i, action):

        self.bearing[i] = (self.bearing[i] + action + 6) % 8
        if self.bearing[i] == 0:
            self.current[i][1] -= 1

        elif self.bearing[i] == 1:
            self.current[i][0] += 1
            self.current[i][1] -= 1

        elif self.bearing[i] == 2:
            self.current[i][0] += 1

        elif self.bearing[i] == 3:
            self.current[i][0] += 1
            self.current[i][1] += 1

        elif self.bearing[i] == 4:
            self.current[i][1] += 1

        elif self.bearing[i] == 5:
            self.current[i][0] -= 1
            self.current[i][1] += 1

        elif self.bearing[i] == 6:
            self.current[i][0] -= 1

        elif self.bearing[i] == 7:
            self.current[i][0] -= 1
            self.current[i][1] -= 1

    def update(self):
        rewards = np.zeros(self.agent_num)
        for i in range(self.sub_num):
            if not self.live[i]:
                continue
            x = self.current[i][0]
            y = self.current[i][1]
            pre_x = self.pre_current[i][0]
            pre_y = self.pre_current[i][1]
            if x == self.target[0] and y == self.target[1]:
                rewards[i] = 50 * (self.gamma ** self.steps)
                self.live[i] = False
                self.info[i] = 1
                self.map[pre_x][pre_y] = 0
                continue
            if x < 0 or y < 0 or x > self.size - 1 or y > self.size - 1:
                rewards[i] = -10
                self.live[i] = False
                self.info[i] = -1
                self.map[pre_x][pre_y] = 0

        for i in range(self.sub_num, self.agent_num):
            if not self.live[i]:
                continue
            x = self.current[i][0]
            y = self.current[i][1]
            pre_x = self.pre_current[i][0]
            pre_y = self.pre_current[i][1]

            if x < 0 or y < 0 or x > self.size - 1 or y > self.size - 1:
                rewards[i] = -10
                self.live[i] = False
                self.map[pre_x][pre_y] = 0
                continue
            conflict_list = self.check_conflict(i)
            if conflict_list:
                rewards[i] = -10
                self.live[i] = False
                self.map[pre_x][pre_y] = 0
                for j in conflict_list:
                    j_x = self.pre_current[j][0]
                    j_y = self.pre_current[j][1]
                    rewards[j] = -10
                    self.map[j_x][j_y] = 0
                continue

            self.map[x][y] = i + 1
            capture_list = self.check_capture(i)
            if capture_list:
                capture_num = len(capture_list)
                rewards[i] = capture_num * 10
                self.info[i] += capture_num
                for j in capture_list:
                    self.info[j] = -4
                    j_x = self.pre_current[j][0]
                    j_y = self.pre_current[j][1]
                    rewards[j] = -10
                    self.map[j_x][j_y] = 0
                continue
            if self.steps >= self.max_steps:
                rewards[i] = -10

        for i in range(self.sub_num):

            if not self.live[i]:
                continue
            x = self.current[i][0]
            y = self.current[i][1]
            pre_x = self.pre_current[i][0]
            pre_y = self.pre_current[i][1]

            if self.mines_map[x][y] == 1:
                rewards[i] = -10
                self.live[i] = False
                self.info[i] = -2
                self.map[pre_x][pre_y] = 0
                continue

            conflict_list = self.check_conflict(i)
            if conflict_list:
                rewards[i] = -10
                self.live[i] = False
                self.info[i] = -3
                self.map[pre_x][pre_y] = 0
                for j in conflict_list:
                    j_x = self.pre_current[j][0]
                    j_y = self.pre_current[j][1]
                    rewards[j] = -10
                    self.info[i] = -3
                    self.map[j_x][j_y] = 0
                continue
            if self.steps >= self.max_steps:
                self.info[i] = -5
                rewards[i] = -10
            self.map[x][y] = i + 1
        return rewards

    def get_states(self):
        sub_states = np.zeros((self.sub_num, self.sub_obs_size))
        opp_states = np.zeros((self.opp_num, self.opp_obs_size))
        this_sonar = np.zeros(5)
        that_sonar = np.zeros(5)

        for i in range(self.sub_num):
            if not self.live[i]:
                continue
            onset = 0
            bearing = self.bearing[i]
            res_pos = [self.target[0] - self.current[i][0], self.target[1] - self.current[i][1]]
            self.get_sonar(i, this_sonar)
            self.get_av(i, that_sonar)
            sub_states[i][onset] = bearing
            onset += 1
            sub_states[i][1] = res_pos[0]
            onset += 1
            sub_states[i][2] = res_pos[1]
            onset += 1
            for j in range(0, 5):
                sub_states[i][j + onset] = this_sonar[j]
                sub_states[i][j + onset + 5] = this_sonar[j]
            onset += 10
            sub_states[i][onset] = self.steps
            onset += 1

            assert onset == self.sub_obs_size

        for i in range(self.opp_num):
            id_in_all = i + self.sub_num
            if not self.live[id_in_all]:
                continue
            onset = 0
            bearing = self.bearing[id_in_all]
            opp_states[i][onset] = bearing
            onset += 1
            for j in range(self.sub_num):
                if not self.live[j]:
                    onset += 2
                    continue
                res_pos = [self.current[j][0] - self.current[i][0], self.current[j][1] - self.current[i][1]]
                opp_states[i][onset] = res_pos[0]
                opp_states[i][onset] = res_pos[1]
                onset += 2
            self.get_av(id_in_all, that_sonar)
            for k in range(5):
                opp_states[i][onset + k] = that_sonar[k]
            onset += 5
            opp_states[i][onset] = self.steps
            onset += 1
            assert onset == self.opp_obs_size

        return merge_array(sub_states, opp_states)

    def check_capture(self, i):
        x = self.current[i][0]
        y = self.current[i][1]
        capture_list = []
        for j in range(self.sub_num):
            if self.live[j]:
                sub_x = self.current[j][0]
                sub_y = self.current[j][1]
                if abs(x - sub_x) <= 1 and abs(y - sub_y) <= 1:
                    self.live[j] = False
                    capture_list.append(j)
        return capture_list

    def check_conflict(self, i):
        conflict_list = []
        x = self.current[i][0]
        y = self.current[i][1]
        if i >= self.sub_num:
            a = self.sub_num
            b = self.agent_num
        else:
            a = 0
            b = self.sub_num

        for j in range(a, b):
            if i == j or self.live[j]:
                continue
            res_x = self.current[j][0]
            res_y = self.current[j][1]
            if res_x == x and res_y == y:
                conflict_list.append(j)
                self.live[j] = False
        return conflict_list

    def get_sonar(self, i, new_sonar):
        x = int(self.current[i][0])
        y = int(self.current[i][1])

        if x < 0 or y < 0:
            for k in range(5):
                new_sonar[k] = 0
            return
        a_sonar = [0.0] * 8

        r = 1
        while y - r >= 0 and self.mines_map[x][(y - r) % self.size] != 1:
            r = r + 1
        a_sonar[0] = 1.0 / r

        r = 1
        while x + r <= self.size - 1 and y - r >= 0 and self.mines_map[(x + r) % self.size][(y - r) % self.size] != 1:
            r = r + 1
        a_sonar[1] = 1.0 / r

        r = 1
        while x + r <= self.size - 1 and self.mines_map[(x + r) % self.size][y] != 1:
            r = r + 1
        a_sonar[2] = 1.0 / r

        r = 1
        while x + r <= self.size - 1 and y + r <= self.size - 1 and self.mines_map[(x + r) % self.size][
            (y + r) % self.size] != 1:
            r = r + 1
        a_sonar[3] = 1.0 / r

        r = 1
        while y + r <= self.size - 1 and self.mines_map[x][(y + r % self.size)] != 1:
            r = r + 1
        a_sonar[4] = (1.0 / r)

        r = 1
        while x - r >= 0 and y + r <= self.size - 1 and self.mines_map[(x - r) % self.size][(y + r) % self.size] != 1:
            r = r + 1
        a_sonar[5] = 1.0 / r

        r = 1
        while x - r >= 0 and self.mines_map[(x - r) % self.size][y] != 1:
            r = r + 1
        a_sonar[6] = 1.0 / r

        r = 1
        while x - r >= 0 and y - r >= 0 and self.mines_map[(x - r) % self.size][(y - r) % self.size] != 1:
            r = r + 1
        a_sonar[7] = 1.0 / r

        for k in range(5):
            new_sonar[k] = a_sonar[(self.bearing[i] + 6 + k) % 8]

    def get_av(self, i, new_sonar):
        x = self.current[i][0]
        y = self.current[i][1]
        a_sonar = [0.0] * 8

        if i < self.sub_num:
            r = 1
            while y - r >= 0 and self.map[x][y - r] == 0 and r <= self.sound:
                r += 1
            a_sonar[0] = 1.0 / r

            r = 1
            while x + r <= self.size - 1 and y - r >= 0 and self.map[x + r][y - r] == 0 and r <= self.sound:
                r += 1
            a_sonar[1] = 1.0 / r

            r = 1
            while x + r <= self.size - 1 and self.map[x + r][y] == 0 and r <= self.sound:
                r += 1
            a_sonar[2] = 1.0 / r

            r = 1
            while x + r <= self.size - 1 and y + r <= self.size - 1 and self.map[x + r][y + r] == 0 and r <= self.sound:
                r += 1
            a_sonar[3] = 1.0 / r

            r = 1
            while y + r <= self.size - 1 and self.map[x][y + r] == 0 and r <= self.sound:
                r += 1
            a_sonar[4] = 1.0 / r

            r = 1
            while x - r >= 0 and y + r <= self.size - 1 and self.map[x - r][y + r] == 0 and r <= self.sound:
                r += 1
            a_sonar[5] = 1.0 / r

            r = 1
            while x - r >= 0 and self.map[x - r][y] == 0 and r <= self.sound:
                r += 1
            a_sonar[6] = 1.0 / r

            r = 1
            while x - r >= 0 and y - r >= 0 and self.map[x - r][y - r] == 0 and r <= self.sound:
                r += 1
            a_sonar[7] = 1.0 / r

        else:
            r = 1
            while y - r >= 0 and self.map[x][y - r] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[0] = 1.0 / r

            r = 1
            while x + r <= self.size - 1 and y - r >= 0 and self.map[x + r][y - r] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[1] = 1.0 / r

            r = 1
            while x + r <= self.size - 1 and self.map[x + r][y] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[2] = 1.0 / r

            r = 1
            while x + r <= self.size - 1 and y + r <= self.size - 1 and self.map[x + r][y + r] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[3] = 1.0 / r

            r = 1
            while y + r <= self.size - 1 and self.map[x][y + r] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[4] = 1.0 / r

            r = 1
            while x - r >= 0 and y + r <= self.size - 1 and self.map[x - r][y + r] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[5] = 1.0 / r

            r = 1
            while x - r >= 0 and self.map[x - r][y] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[6] = 1.0 / r

            r = 1
            while x - r >= 0 and y - r >= 0 and self.map[x - r][y - r] < self.sub_num and r <= self.sound:
                r += 1
            a_sonar[7] = 1.0 / r

        for k in range(5):
            new_sonar[k] = a_sonar[(self.bearing[i] + 6 + k) % 8]

    def who_around(self, agt):
        agent_around = []
        x = self.current[agt][0]
        y = self.current[agt][1]

        for j in range(self.sub_num, self.agent_num):
            a = abs(x - self.current[j][0])
            b = abs(y - self.current[j][1])
            if a <= 3 and b <= 3:
                agent_around.append(j)

        return agent_around

    def judge_move(self, sub, opp, action, predict_action):
        sub_pos = self.virtual_move(sub, action)
        opp_pos = self.virtual_move(opp, predict_action)

        if abs(sub_pos[0] - opp_pos[0]) <= 1 and (sub_pos[1] - opp_pos[1] <= 1) or not sub_pos:
            return False
        return True

    def virtual_move(self, i, action):
        virtual_bearing = (self.bearing[i] + action + 6) % 8
        x = self.current[i][0]
        y = self.current[i][1]

        if virtual_bearing == 0:
            if y > 0:
                y -= 1
            else:
                return False

        elif virtual_bearing == 1:
            if x < self.size - 1 and y > 0:
                x += 1
                y -= 1
            else:
                return False

        elif virtual_bearing == 2:
            if x < self.size - 1:
                x += 1
            else:
                return False

        elif virtual_bearing == 3:
            if x < self.size - 1 and y < self.size - 1:
                x += 1
                y += 1
            else:
                return False

        elif virtual_bearing == 4:
            if y < self.size - 1:
                y += 1
            else:
                return False

        elif virtual_bearing == 5:
            if x > 0 and y < self.size - 1:
                x -= 1
                y += 1
            else:
                return False

        elif virtual_bearing == 6:
            if x > 0:
                x -= 1
            else:
                return False

        elif virtual_bearing == 7:
            if x > 0 and y > 0:
                x -= 1
                y -= 1
            else:
                return False

        return [x, y]

    def check_live(self, i):
        return self.live[i]

    def check_done(self):
        sub_done = not any([self.live[l] for l in range(self.sub_num)])
        done = sub_done
        return done

    def get_info(self):
        return self.info








