import numpy as np
from itertools import combinations
import random
from typing import List

class Agent:    
    def __init__(self, line_length: int, dst_lengths:List[int], dst_nums:List[int]):
        # 钢筋的直径mm
        self.line_size = 32
        # 最大目标钢筋的类别数
        self.max_dst_nums = 10
        # 允许最大废料的个数,状态矩阵的大小为90+10=100
        self.max_unused_nums = 90
        # 原始钢筋的长度
        self.line_length = line_length
        # 目标钢筋的长度
        self.dst_lengths = dst_lengths
        # 目标钢筋的数量
        self.dst_nums = dst_nums
        # 当前钢筋的总数量                        
        self.dst_nums_totle = sum(self.dst_nums)
        # 当前已经使用的钢筋接头数量
        self.used_joints = 0
        # 使用的钢筋数量
        self.used_num = 0
        # 未使用的钢筋长度清单
        self.unused_lines = []
        # 动作执行顺序
        self.action_history = []
        # 动作执行结果
        self.reward = 0
        # 动作执行次数
        self.step_count = 0
        # 可用动作
        self.available_actions = np.zeros(2*self.max_dst_nums, dtype=int)        
        # 是否结束
        self.is_terminal=False
        assert len(self.dst_nums) == len(self.dst_lengths), "输出长度的个数必需和输出种类一致"
        assert len(self.dst_nums) <= self.max_dst_nums, "输出种类个数必需小于最大允许种类" 
        # 参考奖励
        self.ref_reward = self.get_ref_reward()

    # 获得可以执行的动作列表,动作列表的索引为目标钢筋的索引，第一段值表示可以执行裁剪动作，第二段值表示可以执行拼接动作
    def get_actions(self):
        # 可用动作列表 0 无法执行 1 可执行
        self.available_actions = np.zeros(2*self.max_dst_nums, dtype=int) 
        for i in range(len(self.dst_lengths)):
            # 目标钢筋数量为0
            if self.dst_nums[i]==0: 
                self.available_actions[i]=0   # 无法执行裁剪动作
                self.available_actions[self.max_dst_nums+i]=0   # 无法执行拼接动作
                continue
            self.available_actions[i]=1  # 可以裁剪 
            # 目标钢筋长度
            dst_length = self.dst_lengths[i]
            find=False
            # 找到可以拼接的钢筋，即至少有2个钢筋，且长度需要小于目标钢筋长度并且最小尺寸需要大于200
            can_combination = [line for line in self.unused_lines if line>200 and line<dst_length]
            if len(can_combination)<2: continue
            # 枚举所有可能的组合包括2-3个钢筋
            for k in range(2, 4):
                for combination in combinations(can_combination, k):
                    # 组合长度大于等于目标钢筋长度
                    if sum(combination)>=dst_length:
                        find=True
                        break
                if find:
                    break
            if find:
                self.available_actions[self.max_dst_nums+i]=1
        return self.available_actions

    # 执行动作
    def step(self, action):
        # 动作执行顺序
        self.action_history.append(action)
        # 动作执行次数
        self.step_count += 1
        # 目标钢筋索引
        if action<self.max_dst_nums:
            dst_index = action
        else:
            dst_index = action-self.max_dst_nums
        # 目标钢筋长度
        dst_length = self.dst_lengths[dst_index]
        # 裁剪动作
        if action < self.max_dst_nums:
            # 先从未使用的钢筋中找到合适的钢筋
            find = False
            for i in range(len(self.unused_lines)):
                if self.unused_lines[i]>=dst_length:
                    line = self.unused_lines.pop(i)
                    find=True
                    break
            if not find:
                line = self.line_length
                self.used_num += 1

            # 减少未使用的钢筋长度
            cut_size = line - dst_length
            if cut_size>0: self.unused_lines.append(cut_size)                
            # 减少目标钢筋数量
            self.dst_nums[dst_index]-=1
            
            # 如果未使用的钢筋数量超过了最大允许，结束
            if len(self.unused_lines)>self.max_unused_nums:
                self.is_terminal=True
        # 拼接动作
        else:
            find=False
            # 枚举所有可能的组合包括2-3个钢筋
            # 首先需要找到可以拼接的钢筋，即至少有2个钢筋，且长度需要小于目标钢筋长度并且最小尺寸需要大于200
            can_combination = [line for line in self.unused_lines if line>200 and line<dst_length]
            for k in range(2, 4):
                for combination in combinations(can_combination, k):
                    # 组合长度大于等于目标钢筋长度,并且最小尺寸需要大于200
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
        # 目标钢筋数量是否都为0,则结束
        if sum(self.dst_nums)==0:        
            self.is_terminal=True

    # 获得一个参考的奖励
    def get_ref_reward(self):
        unused_lines=[]
        used_joints=0
        line=self.line_length
        # 直接按顺序循环截断或连接
        for i in range(len(self.dst_nums)):
            for j in range(self.dst_nums[i]):
                dst_length=self.dst_lengths[i]
                if line>dst_length:
                    line -= dst_length
                else:
                    if line>0:
                        if line<=200:
                            unused_lines.append(line)
                            line = self.line_length
                        else:
                            line += self.line_length
                            used_joints += 1
                    else:
                        line = self.line_length
                    line -= dst_length
        if line>0:
            unused_lines.append(line)
        unused_line_sum = sum(unused_lines)
        reward = (-self.line_size**2*unused_line_sum/1000*0.00617*2000 - used_joints*10)
        print("参考奖励:",reward,"接头:",used_joints,"剩余:",unused_lines)
        return reward
        
    # 获得奖励    
    def get_reward(self):
        # 奖励函数
        if not self.is_terminal: return 0
        if sum(self.dst_nums)>0:
            return -1
        # 统计未使用的钢筋长度
        unused_line_sum = sum(self.unused_lines)
        # 奖励函数 钢筋直径的平方 * 钢筋长度/1000 * 钢筋重量 * 每吨价格 + 已使用的钢筋接头数量 * 10
        reward = (self.line_size**2*unused_line_sum/1000*0.00617*2000 + self.used_joints*10+self.ref_reward)/self.ref_reward
        # if reward<-1: reward = -1
        return reward
    
    # 是否结束
    def is_done(self):
        return self.is_terminal

    # 获得当前状态
    # 设定原始钢筋的长度为1，则目标钢筋长度为实际长度/原始钢筋的长度
    # shape: (10, 10)
    def get_state(self):
        # 状态为 目标钢筋长度/原始钢筋的长度，目标钢筋数量
        state = np.zeros(self.max_dst_nums + self.max_unused_nums , np.float16)

        # 剔除已经完成的目标钢筋，这里不区分剩余多少数量的钢筋，只区分是否有目标钢筋    
        dst_lengths = sorted([self.dst_lengths[i] for i in range(len(self.dst_lengths)) if self.dst_nums[i]>0])
        for i in range(len(dst_lengths)):
            state[i] = dst_lengths[i]/self.line_size

        # 状态为 剩余钢筋长度/原始钢筋的长度
        for i in range(len(self.unused_lines)):
            state[self.max_dst_nums+i] = self.unused_lines[i]/self.line_length

        return state       

    # 获得id 这里不区分剩余多少数量的钢筋，只区分是否有目标钢筋
    def get_id(self):
        id=0
        for i in range(len(self.unused_lines)):
            id = id*10 + self.unused_lines[i]/self.line_size
        for i in range(len(self.dst_nums)):
            id = id*10 + 1 if self.dst_nums[i]>0 else 0
        return str(id)

if __name__ == '__main__':    
    agent = Agent(line_length=12000, dst_lengths=[4100, 4350, 4700], dst_nums=[552, 658, 462])
    # agent = Agent(line_length=12000, dst_lengths=[4100, 4350, 4700], dst_nums=[8, 5, 3])
    while not agent.is_done():
        actions = agent.get_actions()
        action = random.choice(np.where(actions==1)[0])
        agent.step(action)
            
    reward = agent.get_reward()
    print("奖励：",reward,(1-reward)*agent.ref_reward)
    print("剩余钢筋长度：",agent.unused_lines)
    print("已使用的钢筋接头数量：",agent.used_joints)
    print("已使用的钢筋数量：",agent.used_num)
    