import random
import numpy as np
import copy

def gaussian(x, sigma = 0.25, u = 0) -> float:
    x = x / 10
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2))
    return y

class Maze():
    size = 16
    trace = False
    def __init__(self, agt_num, avTypes):
        
        self.size = 16
        self.numMines = 8
        self.opp_num = 2
        self.sub_num = agt_num - self.opp_num
        self.binarySonar = False
        self.Sound = 8
        self.alpha = 0.0015
        self.LINEAR = 0
        self.EXP = 1
        self.RewardType = self.EXP
        self.opp_range = [0] * self.opp_num

        self.__agent_num = agt_num
        self.__current = []
        self.__prev_current = []
        self.__target = []
        self.__currentBearing = []

        self.__prev_bearing = []
        self.__targetBearing = []
        self.__sonar = []
        self.__av_sonar = []
        self.__range = []
        self.__mines = []
        

        self.__avs = []
        self.__end_state = []
        self.__conflict_state = []
        self.__captured_state = []
        self.refreshMaze(agt_num, avTypes)

    def change_maze(self, agt_num, avTypes, mines):
        self.__agent_num = agt_num
        self.numMines = mines
        self.refreshMaze(agt_num, avTypes)

    def set_conflict(self, i, j):
        self.__end_state[i] = True
        self.__end_state[j] = True
        self.__conflict_state[i] = True
        self.__conflict_state[j] = True

    def set_captured(self, i):
        self.__end_state[i] = True
        self.__captured_state[i] = True

    def check_i_conflict(self, i, avTypes: list):
        if self.__conflict_state[i]:
            return True
        for k in range(self.__agent_num):
            if k == i or avTypes[k] == 2 or self.__end_state[k]:
                continue
            if self.__current[k][0] == self.__current[i][0] and self.__current[k][1] == self.__current[i][1]:
                self.set_conflict(i, k)
                return True
        return False

    def check_i_captured(self, i, avTypes: list):
        if avTypes[i] == 1:
            for k in range(self.__agent_num):
                if k == i or avTypes[k] == 1 or self.__end_state[k]:
                    continue
                res = [0, 0]
                res[0] = self.__current[k][0]
                res[1] = self.__current[k][1]
                if res[0] == self.__current[i][0] and res[1] == self.__current[i][1]:
                    return True
                
                bearings = [self.__currentBearing[k]]
                
                for bearing in bearings:
                    res[0] = self.__current[k][0]
                    res[1] = self.__current[k][1]
                    if bearing == 0:
                        if res[1] > 0:
                            res[1] -= 1

                    elif bearing == 1:
                        if res[0] < self.size - 1 and res[1] > 0:
                            res[0] += 1
                            res[1] -= 1

                    elif bearing == 2:
                        if res[0] < self.size - 1:
                            res[0] += 1

                    elif bearing == 3:
                        if res[0] < self.size - 1 and res[1] < self.size - 1:
                            res[0] += 1
                            res[1] += 1
                    elif bearing == 4:
                        if res[1] < self.size - 1:
                            res[1] += 1

                    elif bearing == 5:
                        if res[0] > 0 and res[1] < self.size - 1:
                            res[0] -= 1
                            res[1] += 1

                    elif bearing == 6:
                        if res[0] > 0:
                            res[0] -= 1

                    elif bearing == 7:
                        if res[0] > 0 and res[1] > 0:
                            res[0] -= 1
                            res[1] -= 1
                    else:
                        pass

                    if res[0] == self.__current[i][0] and res[1] == self.__current[i][1]:
                        return True
                
        else:
            number = 0
            for k in range(self.__agent_num):
                if k == i or avTypes[k] == 2 or self.__end_state[k]:
                    continue
                res = [0, 0]
                res[0] = self.__current[k][0]
                res[1] = self.__current[k][1]
                if res[0] == self.__current[i][0] and res[1] == self.__current[i][1]:
                    self.set_captured(k)
                    number += 1
                    if self.trace:
                        print(f'opp_agent {i}  captrue sub_agent {k}')
                        print(res, self.__current[i])
                else:
                    bearings = [self.__currentBearing[i]]
                    res = [0, 0]
                    for bearing in bearings:
                        res[0] = self.__current[i][0]
                        res[1] = self.__current[i][1]
                        if bearing == 0:
                            if res[1] > 0:
                                res[1] -= 1

                        elif bearing == 1:
                            if res[0] < self.size - 1 and res[1] > 0:
                                res[0] += 1
                                res[1] -= 1

                        elif bearing == 2:
                            if res[0] < self.size - 1:
                                res[0] += 1

                        elif bearing == 3:
                            if res[0] < self.size - 1 and res[1] < self.size - 1:
                                res[0] += 1
                                res[1] += 1
                        elif bearing == 4:
                            if res[1] < self.size - 1:
                                res[1] += 1

                        elif bearing == 5:
                            if res[0] > 0 and res[1] < self.size - 1:
                                res[0] -= 1
                                res[1] += 1

                        elif bearing == 6:
                            if res[0] > 0:
                                res[0] -= 1

                        elif bearing == 7:
                            if res[0] > 0 and res[1] > 0:
                                res[0] -= 1
                                res[1] -= 1
                        else:
                            pass

                        if res[0] == self.__current[k][0] and res[1] == self.__current[k][1]:
                            
                            self.set_captured(k)
                            number += 1
                            if self.trace:
                                print(k, i, res)
                                print(f'opp_agent {i}  captrue sub_agent {k}')
            return number

        return False

    def refreshMaze(self, agt, avTypes):
        if agt < 1:
             self.__agent_num = 1
        elif agt > 10:
             self.__agent_num = 10
        else:
             self.__agent_num = agt

        self.__current = [([0] * 2) for i in range(self.__agent_num)]
        self.__target = [0, 0]
        self.__prev_current = [([0] * 2) for i in range(self.__agent_num)]
        self.__currentBearing = [0] * self.__agent_num
        self.__prev_bearing = [0] * self.__agent_num
        self.__avs = [([0] * self.size) for i in range(self.size)]
        
        self.__mines = [([0] * self.size) for i in range(self.size)]
        self.mine_list = []
        self.__end_state = [False] * self.__agent_num
        self.__conflict_state = [False] * self.__agent_num

        self.__sonar = [([0] * 5) for i in range(self.__agent_num)]
        self.__av_sonar = [([0] * 8) for i in range(self.__agent_num)]
        self.__captured_state = [False] * self.__agent_num
        self.__path = [[] for _ in range(agt)]

        for k in range(self.__agent_num):
            while True:
                if avTypes[k] == 2:
                    x = random.randint(2, self.size - 3)
                    y = random.randint(2, self.size - 3)
                else:
                    x = random.randint(0, self.size - 1)
                    if x >= 2 and x <= self.size - 3:
                        y = random.randint(0,1) + random.randint(0,1) * (self.size - 2)
                    else:
                        y = random.randint(0, self.size - 1)

                self.__current[k][0] = x
                
                self.__current[k][1] = y
                if self.__avs[x][y] == 0:
                    self.__avs[x][y] = k + 1
                    break

            for w in range(2):
                self.__prev_current[k][w] = self.__current[k][w]

            self.__end_state[k] = False
            self.__conflict_state[k] = False
            self.__path[k].append([self.__current[k][0], self.__current[k][1]])
        for i in range(self.opp_num):
            range_list = [0] * self.sub_num
            for j in range(self.sub_num):
                range_list[j] = self.get_i_j_Range(i + self.sub_num, j)
            self.opp_range[i] = sum(range_list)


        while True:
            x = random.randint(2, self.size - 3)
            self.__target[0] = x
            y = random.randint(2, self.size - 3)
            self.__target[1] = y

            if self.__avs[x][y] == 0:
                break
        
        for i in range(self.numMines):
            while True:
                x = random.randint(2, self.size - 3)
                y = random.randint(2, self.size - 3)

                if self.__avs[x][y] == 0 and self.__mines[x][y] == 0 and x != self.__target[0] or y != self.__target[1] :
                    self.__mines[x][y] = 1
                    self.mine_list.append([x, y])
                    break

        for a in range(self.__agent_num):
            self.setCurrentBearing(a, random.randint(0, 7))
            self.__prev_bearing[a] = self.__currentBearing[a]

    def get_i_j_Bearing(self, i, j):
        a = self.__current[i]
        b = self.__current[j]
        return self.get_a_b_Bearing(a, b)
    def get_path(self):
        return self.__path
    def get_a_b_Bearing(self, a, b):
    
        d = [0] * 2
        d[0] = a[0] - b[0]
        d[1] = a[1] - b[1]

        if d[0] == 0 and d[1] < 0:
            return 0  #向上
        if d[0] > 0 and d[1] < 0:
            return 1  #右上
        if d[0] > 0 and d[1] == 0:
            return 2  #右
        if d[0] > 0 and d[1] > 0:
            return 3 #右下
        if d[0] == 0 and d[1] > 0:
            return 4  #下
        if d[0] < 0 and d[1] > 0:
            return 5  #左下
        if d[0] < 0 and d[1] == 0:
            return 6  #左
        if d[0] < 0 and d[1] < 0:
            return 7  #左上
        return 0

    def getTargetBearing(self, i):

        d = [0] * 2
        d[0] = self.__target[0] - self.__current[i][0]
        d[1] = self.__target[1] - self.__current[i][1]

        if d[0] == 0 and d[1] < 0:
            return 0  # 向上
        if d[0] > 0 and d[1] < 0:
            return 1  # 右上
        if d[0] > 0 and d[1] == 0:
            return 2  # 右
        if d[0] > 0 and d[1] > 0:
            return 3  # 右下
        if d[0] == 0 and d[1] > 0:
            return 4  # 下
        if d[0] < 0 and d[1] > 0:
            return 5  # 左下
        if d[0] < 0 and d[1] == 0:
            return 6  # 左
        if d[0] < 0 and d[1] < 0:
            return 7  # 左上
        return 0

    def getCurrentBearing(self, i):
        return self.__currentBearing[i]

    def get_all_CurrentBearing(self):
        return  self.__currentBearing

    def setCurrentBearing(self, i, b):
        self.__currentBearing[i] = b

    def getReward(self, agt: int, pos: list, avTypes):
        x = pos[0]
        y = pos[1]

        if avTypes[agt] == 1:
            if x == self.__target[0] and y == self.__target[1]:
                self.__end_state[agt] = True
                if self.trace:
                    print(f'sub_agent {agt} target')
                return 10.0
            elif x < 0 or y < 0 or x > self.size - 1 or y > self.size - 1:
                self.__end_state[agt] = True
                if self.trace:
                    print(f'sub_agent {agt} out')
                return -5.0
            elif self.__mines[x][y] == 1:
                self.__end_state[agt] = True
                if self.trace:
                    print(f'sub_agent {agt} mine')
                return -10.0
            elif self.check_i_captured(agt, avTypes):
                if self.trace:
                    print(f'sub_agent {agt}  be captured')
                return -10
            elif self.check_i_conflict(agt, avTypes):
                if self.trace:
                    print(f'sub_agent {agt}  be conflict')
                return -10
            return 0
            # turn_reward = 0.0
            # # 4.拐弯惩罚
            # if(self.__currentBearing[agt] != self.__prev_bearing[agt]):
            #     turn_reward = -0.1
            state = self.getState(agt, avTypes)
            # 5. 预防式的靠近惩罚
            close_reward = -max(state[:5]) * 0.1
            # 6.靠近或远离惩罚(应用Reward Shaping方法)
            now_Range = - self.alpha * self.getRange(agt)

            return now_Range + close_reward
        else:

            number = self.check_i_captured(agt, avTypes)
            now_Ranges = [0] * self.sub_num
            for i in range(self.sub_num):
                now_Ranges[i] = self.get_i_j_Range(i, agt)

            new_range = sum(now_Ranges)
            reward = number * 10 + (self.opp_range[agt - self.sub_num] - new_range) * 0.1
            self.opp_range[agt - self.sub_num] = new_range
            return reward

    def get_i_Reward(self, i: int, avTypes):
        return self.getReward(i, self.__current[i], avTypes)

    def get_a_b_Range(self, a, b: list):
        Range = 0
        d = [0] * 2

        d[0] = abs(a[0] - b[0])
        d[1] = abs(a[1] - b[1])
        Range = max(d[0], d[1])
        return Range

    def getRange(self, i):
       return self.get_a_b_Range(self.__current[i], self.__target)

    def get_i_j_Range(self, i, j):
        return self.get_a_b_Range( self.__current[i], self.__current[j])

    def get_all_Range(self):
        k = 0
        all_range = [0] * self.__agent_num
        for k in range(self.__agent_num):
            all_range[k] = self.getRange(k)
        return all_range

    def getTargetRange(self, i):
        return 1.0 / (1 + self.getRange(i))

    def get_all_TargetRange(self):
        Range = [0] * self.__agent_num
        for k in range(self.__agent_num):
            Range[k] = self.getTargetRange(k)
        return Range

    def getSonar(self, agt: int, new_sonar: list):
        x = self.__current[agt][0]
        y = self.__current[agt][1]

        if x < 0 or y < 0:
            for k in range(5):
                new_sonar[k] = 0
            return
        aSonar = [0.0] * 8

        r = 0

        while y - r >= 0 and self.__mines[x][(y-r) % self.size] != 1:
            r = r + 1

        if r == 0:
            aSonar[0] = 2
        else:
            aSonar[0] = 1.0 / r

        r = 0
        while x + r <= self.size - 1 and y - r >= 0 and self.__mines[(x+r) % self.size][(y-r) % self.size] != 1:
            r = r + 1
        if r == 0:
            aSonar[1] = 2
        else:
            aSonar[1] = 1.0 / r

        r = 0
        while x + r <= self.size - 1 and self.__mines[(x+r) % self.size][y] != 1:
            r = r + 1
        if r == 0:
            aSonar[2] = 2
        else:
            aSonar[2] = 1.0 / r

        r = 0
        while x + r <= self.size - 1 and y + r <= self.size - 1 and self.__mines[(x+r) % self.size][(y+r) % self.size] != 1:
            r = r + 1
        if r == 0:
            aSonar[3] = 2
        else:
            aSonar[3] = 1.0 / r

        r = 0
        while y + r <= self.size - 1 and self.__mines[x][(y+r % self.size)] != 1:
            r = r + 1
        if r == 0:
            aSonar[4] = 2
        else:
            aSonar[4] = (1.0 / r)

        r = 0
        while x-r >= 0 and y+r <= self.size-1 and self.__mines[(x-r) % self.size][(y+r) % self.size] != 1:
            r = r + 1
        if r == 0:
            aSonar[5] = 2
        else:
            aSonar[5] = 1.0 / r
        r = 0
        while x-r >= 0 and self.__mines[(x-r) % self.size][y] != 1:
            r = r + 1
        if r == 0:
            aSonar[6] = 2
        else:
            aSonar[6] = 1.0 / r

        r = 0
        while x-r >= 0 and y-r >= 0 and self.__mines[(x-r) % self.size][(y-r) % self.size] != 1:
            r = r + 1
        if r == 0:
            aSonar[7] = 2
        else:
            aSonar[7] = 1.0 / r

        self.__currentBearing = self.get_all_CurrentBearing()

        for k in range(5):
            
            new_sonar[k] = aSonar[(self.__currentBearing[agt] + 6 + k) % 8]
            if self.binarySonar:
                if new_sonar[k] < 1:
                    new_sonar[k] = 0
        return

    def getAVSonar(self, agt: int, new_av_sonar: list):

        x = self.__current[agt][0]
        y = self.__current[agt][1]
        aSonar = [0] * 8
        r = 0

        if agt < self.sub_num:
            while y - r >= 0 and (self.__avs[x][y - r] == agt + 1 or self.__avs[x][y - r] == 0) and r <= self.Sound:
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x + r <= self.size - 1 and y - r >= 0 and (
                    self.__avs[x + r][y - r] == agt + 1 or self.__avs[x + r][y - r] == 0) and r <= self.Sound:  # 右上
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x + r <= self.size - 1 and (
                    self.__avs[x + r][y] == agt + 1 or self.__avs[x + r][y] == 0) and r <= self.Sound:  # 右侧
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x + r <= self.size - 1 and y + r <= self.size - 1 and (
                    self.__avs[x + r][y + r] == agt + 1 or self.__avs[x + r][y + r] == 0) and r <= self.Sound:  # 右下
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while y + r <= self.size - 1 and (
                    self.__avs[x][y + r] == agt + 1 or self.__avs[x][y + r] == 0) and r <= self.Sound:  # 下
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x - r >= 0 and y + r <= self.size - 1 and (
                    self.__avs[x - r][y + r] == agt + 1 or self.__avs[x - r][y + r] == 0) and r <= self.Sound:  # 左下
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x - r >= 0 and (self.__avs[x - r][y] == agt + 1 or self.__avs[x - r][y] == 0) and r <= self.Sound:  # 左
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x - r >= 0 and y - r >= 0 and (
                    self.__avs[x - r][y - r] == agt + 1 or self.__avs[x - r][y - r] == 0) and r <= self.Sound:
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r
        else:
            while y - r >= 0 and (self.__avs[x][y - r] > self.sub_num or self.__avs[x][y - r] == 0) and r <= self.Sound:
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or y - r < 0:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x + r <= self.size - 1 and y - r >= 0 and (
                    self.__avs[x + r][y - r] > self.sub_num or self.__avs[x + r][y - r] == 0) and r <= self.Sound:  # 右上
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or x + r > self.size - 1 or y - r < 0:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x + r <= self.size - 1 and (
                    self.__avs[x + r][y] > self.sub_num or self.__avs[x + r][y] == 0) and r <= self.Sound:  # 右侧
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or x + r > self.size - 1:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x + r <= self.size - 1 and y + r <= self.size - 1 and (
                    self.__avs[x + r][y + r] > self.sub_num or self.__avs[x + r][y + r] == 0) and r <= self.Sound:  # 右下
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or x + r > self.size - 1 or y + r > self.size - 1:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while y + r <= self.size - 1 and (
                    self.__avs[x][y + r] > self.sub_num or self.__avs[x][y + r] == 0) and r <= self.Sound:  # 下
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or y + r > self.size - 1:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x - r >= 0 and y + r <= self.size - 1 and (
                    self.__avs[x - r][y + r] > self.sub_num or self.__avs[x - r][y + r] == 0) and r <= self.Sound:  # 左下
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or x - r < 0 and y + r > self.size - 1:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x - r >= 0 and (self.__avs[x - r][y] > self.sub_num or self.__avs[x - r][y] == 0) and r <= self.Sound:  # 左
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or x - r < 0:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

            r = 0
            while x - r >= 0 and y - r >= 0 and (
                    self.__avs[x - r][y - r] > self.sub_num or self.__avs[x - r][y - r] == 0) and r <= self.Sound:
                r += 1
            if r == 0:
                aSonar[0] = 2
            elif r > self.Sound or x - r < 0 or y - r < 0:
                aSonar[0] = 0.0
            else:
                aSonar[0] = 1.0 / r

        for k in range(5):
            new_av_sonar[k] = aSonar[(self.__currentBearing[agt] + 6 + k) % 8]
            if self.binarySonar and new_av_sonar[k] < 1:
                new_av_sonar[k] = 0
        

    def virtual_move(self, agt: int, d: int):
        bearing = d

        res = [0,0]

        res[0] = self.__current[agt][0]
        res[1] = self.__current[agt][1]


        if bearing == 0:
            if res[1] > 0 :
                res[1] -= 1

        elif bearing == 1:
            if res[0] < self.size - 1 and res[1] > 0:
                res[0] += 1
                res[1] -= 1

        elif bearing == 2:
            if res[0] < self.size - 1:
                res[0] += 1

        elif bearing == 3:
            if res[0] < self.size - 1 and res[1] < self.size - 1:

                res[0] += 1
                res[1] += 1
        elif bearing == 4:
            if res[1] < self.size - 1:
                res[1] += 1

        elif bearing == 5:
            if res[0] > 0 and res[1] < self.size - 1:
                res[0] -= 1
                res[1] += 1

        elif bearing == 6:
            if res[0] > 0:
                res[0] -= 1

        elif bearing == 7:
            if res[0] > 0 and res[1] > 0:
                res[0] -= 1
                res[1] -= 1
        else:
            pass

        return res


    def JudgeMove(self, sub, opp,action, predict_action):
        agt_bearing = (self.__currentBearing[sub] + action + 8) % 8
        agt_res = self.virtual_move(sub, agt_bearing)
        opp_bearing = (self.__currentBearing[opp] + predict_action + 8) % 8
        opp_res = self.virtual_move(opp, opp_bearing) 

        if agt_res[0] == opp_res[0] and agt_res[1] == opp_res[1]:
            return False
        
        bearings = [opp_bearing]
        res = [0,0]
        for bearing in bearings:
            res[0] = opp_res[0]
            res[1] = opp_res[1]
            if bearing == 0:
                if res[1] > 0:
                    res[1] -= 1

            elif bearing == 1:
                if res[0] < self.size - 1 and res[1] > 0:
                    res[0] += 1
                    res[1] -= 1

            elif bearing == 2:
                if res[0] < self.size - 1:
                    res[0] += 1

            elif bearing == 3:
                if res[0] < self.size - 1 and res[1] < self.size - 1:
                    res[0] += 1
                    res[1] += 1
            elif bearing == 4:
                if res[1] < self.size - 1:
                    res[1] += 1

            elif bearing == 5:
                if res[0] > 0 and res[1] < self.size - 1:
                    res[0] -= 1
                    res[1] += 1

            elif bearing == 6:
                if res[0] > 0 :
                    res[0] -= 1

            elif bearing == 7:
                if res[0] > 0 and res[1] > 0:
                    res[0] -= 1
                    res[1] -= 1
            else:
                pass

            if res[0] == agt_res[0] and res[1] == agt_res[1]: 
                #print('out',sub, action + 2, opp, agt_res, opp_res, res)
                return False
        #print('maze',sub, action + 2, opp, agt_res, opp_res, res)
        return True

    def getState(self, agt: int, avType):
        this_Sonar = [0] * 5
        that_Sonar = [0] * 5
        this_AVSonar = [0] * 5
        that_AVSonar = [0] * 5
        bearings = [0]*8

        self.getSonar(agt, that_Sonar)
        self.getAVSonar(agt, that_AVSonar)

        for i in range(5):
            this_Sonar[i] = that_Sonar[i]
            this_AVSonar[i] = that_AVSonar[i]

        for i in range(8):
            bearings[i] = 0.0

        if avType[agt] == 2:
            for k in range(self.sub_num):
                this_bearing = (8 + self.get_i_j_Bearing(k, agt) - self.getCurrentBearing(agt)) % 8  #未来的方位减去现在的方位
                this_range = self.get_i_j_Range(k, agt)
                if this_range < bearings[this_bearing] or bearings[this_bearing] == 0:
                    bearings[this_bearing] = this_range
            return np.hstack((this_AVSonar, bearings))
        else:
            relative_position = [self.__target[0] - self.__current[agt][0], self.__target[1] - self.__current[agt][1]]
            this_bearing = (8 + self.getTargetBearing(agt) - self.getCurrentBearing(agt)) % 8
            this_targetRange = self.getTargetRange(agt)
            bearings[this_bearing] = this_targetRange
            return np.hstack((this_Sonar, this_AVSonar, relative_position, bearings))

    def get_all_State(self, avTypes):
        state = [0 for i in range(self.__agent_num)]
        for agt in range(self.__agent_num):
            state[agt] = self.getState(agt, avTypes)
            #print(f'agt:{agt}, pos{self.__current[agt]}')
        return state

    def move(self, i, d):
        in_flag = False
        for k in range(2):
            self.__prev_current[i][k] = self.__current[i][k]

        self.__prev_bearing[i] = self.__currentBearing[i]

        self.__currentBearing[i] = (self.__currentBearing[i] + d + 8) % 8
        #print('agt',i, self.__currentBearing[i])

        if self.__currentBearing[i] == 0:
            self.__current[i][1] -= 1

        elif self.__currentBearing[i] == 1:
            self.__current[i][0] += 1
            self.__current[i][1] -= 1

        elif self.__currentBearing[i] == 2:
            self.__current[i][0] += 1

        elif self.__currentBearing[i] == 3:
            self.__current[i][0] += 1
            self.__current[i][1] += 1

        elif self.__currentBearing[i] == 4:
            self.__current[i][1] += 1

        elif self.__currentBearing[i] == 5:
            self.__current[i][0] -= 1
            self.__current[i][1] += 1

        elif self.__currentBearing[i] == 6:
            self.__current[i][0] -= 1

        elif self.__currentBearing[i] == 7:
            self.__current[i][0] -= 1
            self.__current[i][1] -= 1

        if 0 <= self.__current[i][0] < self.size and 0 <= self.__current[i][1] < self.size:
            in_flag = True
        if not in_flag:
            self.__current[i][0] = self.__prev_current[i][0]
            self.__current[i][1] = self.__prev_current[i][1]
        self.__path[i].append([self.__current[i][0], self.__current[i][1]])


    def endState(self, agt, avTypes):

        x = self.__current[agt][0]
        y = self.__current[agt][1]
        if avTypes[agt] == 2:
            return False
        if self.__end_state[agt]:
            return True

        elif self.__conflict_state[agt]:

            self.__end_state[agt] = True
            return self.__end_state[agt]
        
        elif self.__captured_state[agt]:
            self.__end_state[agt] = True
            return self.__end_state[agt]

        elif x < 0 or y < 0:
            self.__end_state[agt] = True
            return self.__end_state[agt]

        elif x == self.__target[0] and y == self.__target[1]:
            self.__end_state[agt] = True
            return self.__end_state[agt]

        elif self.__mines[x][y] == 1 or self.check_i_conflict(agt, avTypes) or self.check_i_captured(agt, avTypes) or self.__end_state[agt]:# 踩雷or检测冲突oragt已经停止
            self.__end_state[agt] = True

        return False

    def checkendState(self, agt):
        return self.__end_state[agt]

    def endState_all(self, avTypes): 

        bl = True
        for k in range(self.__agent_num):
            if avTypes[k] == 2:
                continue
            if not self.__end_state[k]:
                bl = False
                break
        return bl

    def who_around(self, agt, agent_type):
        agent_around = []
        res = [0, 0]
        res[0] = self.__current[agt][0]
        res[1] = self.__current[agt][1]
        for j in range(self.sub_num, self.__agent_num):
            a = abs(res[0] - self.__current[j][0])
            b = abs(res[1] - self.__current[j][1])
            if a <= 3 and b <= 3:
                agent_around.append(j)

        return agent_around 
    
    def find_mistake(self, a):
        for i in range(self.sub_num, self.__agent_num):
            if self.__avs[self.__current[i][0]][self.__current[i][1]] == 0:
                print(i, 'mistake', a)
                exit(0)
    def clear_av(self):
        for i in range(len(self.__avs)):
            for j in range(len(self.__avs)):
                self.__avs[i][j] = 0

    def set_av(self, agt):
        self.__avs[self.__current[agt][0]][self.__current[agt][1]] = agt + 1
    
    def get_current(self):
        return self.__current
    def get_prev_current(self):
        return self.__prev_current
    def get_target(self):
        return self.__target
    def get_currentBearing(self):
        return self.__currentBearing
    def get_prev_bearing(self):
        return self.__prev_bearing
    def get_targetBearing(self):
        return self.__targetBearing
    def get_sonar(self):
        return self.__sonar
    def get_av_sonar(self):
        return self.__av_sonar
    def get_range(self):
        return self.__range
    def get_mines(self):
        return self.__mines
    def get_avs(self):
        return self.__avs
    def get_end_state(self):
        return self.__end_state
    def get_conflict_state(self):
        return self.__conflict_state
    def get_captured_state(self):
        return self.__captured_state
    def get_mine_list(self):
        return self.mine_list
    def assign_maze(self, m):

        self.__current = copy.deepcopy(m.get_current())
        self.__prev_current = copy.deepcopy(m.get_prev_current())
        self.__target = copy.deepcopy(m.get_target())
        self.__currentBearing = copy.deepcopy(m.get_currentBearing())
        self.__prev_bearing = copy.deepcopy(m.get_prev_bearing())

        self.__sonar = copy.deepcopy(m.get_sonar())
        self.__av_sonar = copy.deepcopy(m.get_av_sonar())
        self.__range = copy.deepcopy(m.get_range())
        self.__mines = copy.deepcopy(m.get_mines())
        self.__avs = copy.deepcopy(m.get_avs())
        self.__end_state = copy.deepcopy(m.get_end_state())
        self.__conflict_state = copy.deepcopy(m.get_conflict_state())
        self.__captured_state = copy.deepcopy(m.get_captured_state())

