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
        self.N: Dict[int, np.ndarray] = {}
        self.Q: Dict[int, np.ndarray] = {}
        self.P: Dict[int, np.ndarray] = {}

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
        return self.N[s] / np.sum(self.N[s])  

    def run_simulation(self, agent):
        s = agent.get_id()

        if s not in self.P:
            # 叶子展开
            # p, v = self.nnet.predict(agent)
            p = agent.get_actions().copy()
            p = p/np.sum(p) 
            v = 0
            self.P[s] = p
            self.N[s] = np.zeros(len(p), dtype=int)
            self.Q[s] = np.zeros(len(p), dtype=float)
            return v
        
        # 选择当前最高得分的动作
        q = self.Q[s] + self.cpuct * self.P[s] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s])
        availables = agent.get_actions()
        nz_idx = np.nonzero(availables)
        a = nz_idx[0][np.argmax(q[nz_idx])]

        agent.step(a) # 下一步

        # 结束，计算最终奖励
        if agent.step_count - self.start_step_count >= self.max_num_steps or agent.is_done():
            v = agent.get_reward()
        else:
            v = self.run_simulation(agent)

        # 更新步骤
        self.N[s][a] += 1                                                                           
        self.Q[s][a] += (v - self.Q[s][a]) / self.N[s][a]
        return v
            
if __name__ == "__main__":
    # agent = Agent(line_length=12000, dst_lengths=[4100, 4350, 4700], dst_nums=[852, 660, 162])
    agent = Agent(line_length=12000, dst_lengths=[4100, 4350, 4700], dst_nums=[8, 5, 3])
    mcts = MCTS(num_simulations=10000, max_num_steps=1000, cpuct=5)
    while not agent.is_done():
        actions = agent.get_actions()
        act_probs = mcts.policy(agent)
        ACTONS_LEN = len(act_probs)
        action = np.random.choice(range(ACTONS_LEN), p=act_probs) 
        agent.step(action)
            
    reward = agent.get_reward()
    print("奖励：",reward)
    print("剩余钢筋长度：",agent.unused_lines)
    print("已使用的钢筋接头数量：",agent.used_joints)
    print("已使用的钢筋数量：",agent.used_num)           