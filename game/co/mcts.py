from agent import Agent
from typing import *
import numpy as np
import copy

class MCTS:
    """蒙特卡洛树搜索
        Attributes:
            nnet : 神经网络
            num_simulations : 模拟次数
            max_num_steps: 最大探索深度
            cpuct : 探索因子
    """
    def __init__(
            self,
            num_simulations: int,
            max_num_steps: int,
            cpuct: float = 1.0,
    ):
        self.num_simulations = num_simulations
        self.max_num_steps = max_num_steps
        self.cpuct = cpuct

        # N 访问次数 Q 平均回报 P 动作概率
        self.N: Dict[str, np.ndarray] = {}
        self.Q: Dict[str, np.ndarray] = {}
        self.P: Dict[str, np.ndarray] = {}

    def policy(self, agent: Agent) -> np.ndarray:
        """ 按 num_simulations 次数模拟，返回动作的概率，即策略
        Args:
            state: 当前状态
        Returns:
            probs: 返回探索次数占比
        """
        self.start_step_count = agent.step_count
        for _ in range(self.num_simulations):
            _agent = copy.deepcopy(agent)
            self.run_simulation(_agent)

        s = agent.get_id()
        # print(self.Q[s])
        return self.N[s] / np.sum(self.N[s])  

    def run_simulation(self, agent):
        state_stack = []
        action_stack = []
        reward_stack = []
        totel_reward = 0
        while True:           
            current_state = agent.get_id()
            if current_state not in self.P:
                # 叶子展开
                p = agent.get_actions().copy()
                p = p / np.sum(p)
                v = 0
                self.P[current_state] = p
                self.N[current_state] = np.zeros(len(p), dtype=int)
                self.Q[current_state] = np.zeros(len(p), dtype=float)
                totel_reward += v
                break
            
            # 选择当前最高得分的动作
            q = self.Q[current_state] + self.cpuct * self.P[current_state] * np.sqrt(np.sum(self.N[current_state])) / (1 + self.N[current_state])
            availables = agent.get_actions()
            nz_idx = np.nonzero(availables)
            action = nz_idx[0][np.argmax(q[nz_idx])]

            state_stack.append(current_state)
            action_stack.append(action)
            reward_stack.append(totel_reward)

            agent.step(action)  # 下一步

            # 结束，计算最终奖励
            if agent.step_count - self.start_step_count >= self.max_num_steps or agent.is_done():
                reward = agent.get_reward()
                totel_reward += reward
                break

        # 更新步骤
        while state_stack:
            state = state_stack.pop()
            action = action_stack.pop()
            reward = totel_reward - reward_stack.pop()
            self.N[state][action] += 1
            self.Q[state][action] += (reward - self.Q[state][action]) / self.N[state][action]
            
if __name__ == "__main__":
    # agent = Agent(line_length=12000, dst_lengths=[4100, 4350, 4700], dst_nums=[552, 658, 462])
    agent = Agent(line_length=12000, dst_lengths=[4100, 4350, 4700], dst_nums=[55, 65, 46])
    mcts = MCTS(num_simulations=200, max_num_steps=10000, cpuct=5)
    all_dst_nums=agent.dst_nums_totle    
    while not agent.is_done():
        if agent.step_count%100==0:
            print(agent.step_count, '%.2f%%'%(agent.step_count/agent.dst_nums_totle*100))
        actions = agent.get_actions()
        act_probs = mcts.policy(agent)
        ACTONS_LEN = len(act_probs)
        action = np.random.choice(range(ACTONS_LEN), p=act_probs) 
        agent.step(action)
            
    reward = agent.get_reward()
    print("奖励：",reward,reward*agent.ref_reward)
    print("剩余钢筋长度：",agent.unused_lines)
    print("已使用的钢筋接头数量：",agent.used_joints)
    print("已使用的钢筋数量：",agent.used_num)           