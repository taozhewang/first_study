import numpy as np
from itertools import combinations
import random

class Agent:    
    def __init__(self, line_size:int, line_length: int, dst_lengths: list[int], dst_nums:list[int]):
        # 钢筋的直径mm
        self.line_size = line_size
        # 原始钢筋的长度
        self.line_length = line_length
        # 目标钢筋的长度
        self.dst_lengths = dst_lengths
        # 目标钢筋的数量
        self.dst_nums = dst_nums
        # 当前已经使用的钢筋接头数量
        self.used_joints = 0
        # 使用的钢筋数量
        self.used_num = 0
        # 未使用的钢筋长度清单
        self.unused_lines = []
        # 动作空间大小（裁剪和拼接）
        self.action_space = 2 * len(self.dst_lengths)
        # 动作执行顺序
        self.action_history = []
        # 动作执行结果
        self.reward = 0
        # 动作执行次数
        self.step_count = 0

    # 获得可以走的动作列表
    def get_actions(self):
        # 可用动作列表 0 无法执行 1 可执行
        available_actions = np.ones((2, len(self.dst_lengths)), dtype=int)
        for i in range(len(self.dst_lengths)):
            # 目标钢筋数量为0
            if self.dst_nums[i]==0: 
                available_actions[0][i]=0   # 无法执行裁剪动作
                available_actions[1][i]=0   # 无法执行拼接动作
                continue 
            # 目标钢筋数量不足
            if len(self.unused_lines)<2:
                available_actions[1][i]=0   # 无法执行拼接动作
                continue
            # 目标钢筋长度
            dst_length = self.dst_lengths[i]
            find=False
            # 枚举所有可能的组合包括2-3个钢筋
            for k in range(2, 4):
                for combination in combinations(self.unused_lines, k):
                    # 组合长度大于等于目标钢筋长度
                    if sum(combination)>=dst_length:
                        find=True
                        break
                if find:
                    break
            if not find:
                available_actions[1][i]=0

        # 获取可用的动作坐标
        indices = np.where(available_actions == 1)
        # 将索引转换为坐标
        action_coordinates = list(zip(indices[0], indices[1]))
        return action_coordinates

    # 执行动作
    def step(self, action):
        # 动作执行顺序
        self.action_history.append(action)
        # 动作执行次数
        self.step_count += 1
        # 目标钢筋索引
        dst_index = action[1]
        dst_length = self.dst_lengths[dst_index]
        # 裁剪动作
        if action[0] == 0:
            if len(self.unused_lines)==0 or self.unused_lines[-1]<dst_length:
                self.unused_lines.append(self.line_length)
                self.used_num += 1            
            cut_size = self.unused_lines[-1] - dst_length
            if cut_size==0: self.unused_lines.pop()
            else: self.unused_lines[-1] = cut_size
            # 减少目标钢筋数量
            self.dst_nums[dst_index]-=1
        # 拼接动作
        else:
            find=False
            # 枚举所有可能的组合包括2-3个钢筋
            for i in range(2, 4):
                for combination in combinations(self.unused_lines, i):
                    # 组合长度大于等于目标钢筋长度
                    combination_size = sum(combination)
                    if combination_size>=dst_length:
                        find=True
                        # 减少目标钢筋数量
                        self.dst_nums[dst_index]-=1
                        # 增加已使用的钢筋接头数量
                        self.used_joints+=len(combination)-1
                        # 减少未使用的钢筋长度
                        for c in combination:
                            self.unused_lines.remove(c)
                        if combination_size-dst_length>0:
                            self.unused_lines.append(combination_size-dst_length)                            
                        break
                if find:
                    break

    # 获得奖励    
    def get_reward(self):
        # 奖励函数
        # 统计未使用的钢筋长度
        unused_line_sum = sum(self.unused_lines)
        # 奖励函数
        reward = -unused_line_sum*self.line_size**2*0.00617*2000 - self.used_joints*10       
        return reward
    
    def is_done(self):
        # 目标钢筋数量是否都为0
        if sum(self.dst_nums)==0:
            return True
        return False

if __name__ == '__main__':    
    agent = Agent(line_size=32, line_length=12000, dst_lengths=[4100, 4350, 4700], dst_nums=[852, 660, 162])

    while not agent.is_done():
        actions = agent.get_actions()
        action = random.choice(actions)
        agent.step(action)
        # print(action)
            
    reward = agent.get_reward()
    print("奖励：",reward)
    print("剩余钢筋长度：",agent.unused_lines)
    print("已使用的钢筋接头数量：",agent.used_joints)
    print("已使用的钢筋数量：",agent.used_num)
    