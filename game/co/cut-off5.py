#%%
import numpy as np
import random

from core import pattern_oringin, calc_loss_joint, calc_cost, calc_completion_lenghts

'''
用蚁群算法求解钢筋切割问题
'''

# 原始钢筋长度
l = 12000
# 最小可接长度，小于200的部分会做为废料
l_min = 200
# 钢筋的规格
l_size = 32
# 目标钢筋长度
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# 目标钢筋的数量
need = np.array([552, 658, 462], dtype=int)

# 初始化单个组合的最大数量
max_num = 1
# 最大的组合长度
radius = 10
# 组合数最小余料
losses1 = 30

# 蚁群算法参数
# 最大循环次数
max_iterations = 1000000
# 蚂蚁数量
ant_count = 50  
# 信息素持久因子
rho = 0.5  
# 信息素重要程度因子
alpha = 0.1 
# 启发式因子 
beta = 2  

# 评估函数
def evaluate(solution, need, patterns):
    hascut_lengths = np.zeros_like(need)
    cost = 0
    for i in patterns:
        hascut_lengths += patterns[i][0]*solution[i]
        cost += patterns[i][3]*solution[i]

    # 如果组合的长度不足以切割目标钢筋，这里多匹配和少匹配都算到里面
    bar_lengths = need - hascut_lengths
    # 计算尾料的成本
    dl=np.array(list(L.values()))
    loss, joint = calc_loss_joint(bar_lengths, l, dl, l_min)
    cost += calc_cost(loss, joint, l_size)    
    # 计算成本和完成距离目标的距离
    cost += np.sum(np.abs(bar_lengths))*1000
    return cost

# 定义蚂蚁类
class Ant:
    def __init__(self, patterns, need):
        self.path = []  # 解决方案路径
        self.cost = 0   # 成本
        self.patterns = patterns
        self.patterns_length = len(patterns)
        self.need = need

    # 计算剩余长度的路径ID
    def cut_off_to_rod_length(self, has_cut_off):
        return np.sum([has_cut_off[i] * (10**i) for i in range(len(has_cut_off))])

    # 构建解决方案
    def construct_solution(self, pheromone, heuristic):
        solution = np.zeros(self.patterns_length, dtype=int)
        has_cut_off=np.copy(self.need)
        while np.any(has_cut_off > 0):
            loa_lengths = self.cut_off_to_rod_length(has_cut_off)

            probabilities = [(pheromone[loa_lengths][i] ** alpha) * (heuristic[i] ** beta) for i in range(self.patterns_length)]
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            choice = np.random.choice(range(self.patterns_length), p=probabilities)

            # 如果此路不通，标记信息素为0
            has_cut_off -= self.patterns[choice][0]
            if np.any(has_cut_off < 0):
                pheromone[loa_lengths][choice]=0
                # 如果所有的路都不通，则上一步的信息素置0
                if np.all(pheromone[loa_lengths] == 0):
                    _loa_lengths, _choice = self.path[-1]
                    pheromone[_loa_lengths][_choice] = 0
                break    

            solution[choice] += 1
            self.path.append((loa_lengths, choice))            

        self.cost = evaluate(solution, self.need, self.patterns)

# 求各种组合的列表
patterns = pattern_oringin(l, L, losses1, radius)
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")# 产生patterns，最低1个组合，因为需要处理尾料

# 路径的长度，也就是状态的数量，这里不允许超量切割，所以是钢筋的剩余状态hash数量 
rod_length = np.sum([need[i] * (10**i) for i in range(len(need))])
# 初始化信息素矩阵 从一个状态到另外一个状态的概率
pheromone = np.ones((rod_length + 1, patterns_length + 1))
# 初始化启发式信息，这个是个常数，不用更新，表示的是路径之间的相关度
heuristic = [1 / patterns_length for _ in range(patterns_length)]        

# 主循环
for iteration in range(100):
    ants = [Ant(patterns, need) for _ in range(ant_count)]

    # 构建解决方案
    for ant in ants:
        ant.construct_solution(pheromone, heuristic)

    # 更新信息素
    best_cost = min(ant.cost for ant in ants)
    best_ants = [ant for ant in ants if ant.cost == best_cost]
    for ant in best_ants:
        for rod_length, length in ant.path:
            pheromone[rod_length][length] += 1 / ant.cost
    pheromone *= rho

    # 输出最优解
    best_ant = min(ants, key=lambda ant: ant.cost)
    best_solution = np.zeros(patterns_length, dtype=int)
    for rod_length, length in best_ant.path:
        best_solution[length] += 1
    best_used = calc_completion_lenghts(best_solution, need, patterns)
    print(f"{iteration}: 最佳成本: {best_ant.cost} 最佳路径: {best_used} 目标: {need}",)
