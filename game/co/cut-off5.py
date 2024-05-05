#%%
import numpy as np
import random

from core import pattern_oringin_by_sampling, calc_cost_by_unmatched, calc_completion_lenghts

'''
用蚁群算法求解钢筋切割问题

废料长度: 227100
接头数量: 418
总成本: 2873851.936
'''

# 原始钢筋长度
l = 12000
# 最小可接长度，小于200的部分会做为废料
l_min = 200
# 钢筋的规格
l_size = 32
# 目标钢筋长度
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
L_values = np.array(list(L.values()))
# 目标钢筋的数量
need = np.array([552, 658, 462], dtype=int)

# 初始化单个组合的最大数量
max_num = 1
# 最大的组合长度
radius = 10
# 组合的采样数量
sampling_count = 5000

# 蚁群算法参数
# 最大循环次数
max_iterations = 1000000
# 蚂蚁数量
ant_count = 200  
# 信息素持久因子
rho = 0.5  
# 信息素重要程度因子
alpha = 1 
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
    cost += calc_cost_by_unmatched(bar_lengths, l, L_values, l_size)
    return cost

# 定义蚂蚁类
class Ant:
    def __init__(self, patterns, need, max_pattern_length):
        self.path = []  # 解决方案路径
        self.cost = 0   # 成本
        self.patterns = patterns
        self.patterns_length = len(patterns)
        self.need = need
        self.max_pattern_length = max_pattern_length

    # 计算剩余长度的路径ID, 这里将超过pattern长度的部分都归为一个路径
    def cut_off_to_rod_length(self, has_cut_off):
        return np.sum([(has_cut_off[i]%max_pattern_length) * (10**i) for i in range(len(has_cut_off))])

    # 构建解决方案
    def construct_solution(self, pheromone, heuristic):
        solution = np.zeros(self.patterns_length, dtype=int)
        has_cut_off=np.copy(self.need)

        patterns_idxs = list(range(self.patterns_length))
        while np.any(has_cut_off > 0):            
            loa_lengths = self.cut_off_to_rod_length(has_cut_off)
            if len(self.path) == 0:
                # 第一步随机选择路径
                choice = random.choice(patterns_idxs)
            else:
                # 计算路径的启发式信息
                probabilities = pheromone[loa_lengths]**alpha * heuristic
                probabilities = probabilities/np.sum(probabilities)
                # 选择路径
                choice = np.random.choice(patterns_idxs, p=probabilities)

            # 如果此路不通，标记信息素为0
            has_cut_off -= self.patterns[choice][0]
            if np.any(has_cut_off < 0) :
                pheromone[loa_lengths][choice]=0
                break    

            solution[choice] += 1
            self.path.append((loa_lengths, choice))            
        # print(np.var(probabilities))
        self.cost = evaluate(solution, self.need, self.patterns)

# 求各种组合的列表
patterns = pattern_oringin_by_sampling(l, L, sampling_count, radius)
patterns_length = len(patterns)
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")# 产生patterns，最低1个组合，因为需要处理尾料
patterns_lengths = np.array([patterns[i][0] for i in range(patterns_length)])
max_pattern_length = np.max(patterns_lengths)

# 路径的长度，也就是状态的数量，这里不允许超量切割，所以是钢筋的剩余状态hash数量 
rod_length = np.sum([max_pattern_length * (10**i) for i in range(len(need))])
# 初始化信息素矩阵 从一个状态到另外一个状态的概率
pheromone = np.ones((rod_length+1, patterns_length))

# 初始化启发式信息，这个是个常数，不用更新，表示的是路径之间的相关度
heuristic = (np.ones(patterns_length)/patterns_length)**beta

# 主循环
nochange_count = 0
best_cost = np.inf
best_solution = None
best_used = None
avg_cost = 0
for iteration in range(10000):
    ants = [Ant(patterns, need, max_pattern_length) for _ in range(ant_count)]

    # 构建解决方案
    for ant in ants:
        ant.construct_solution(pheromone, heuristic)

    # 计算当前蚂蚁的最优解
    costs = np.array([ant.cost for ant in ants])
    curr_min_idx = np.argmin(costs)
    curr_best_cost = costs[curr_min_idx]
    curr_avg_cost = np.mean(costs)
    # 更新平均成本
    avg_cost = avg_cost*0.9 + curr_avg_cost*0.1
    nochange_count +=1

    # 更新最优解
    if curr_best_cost < best_cost:
        solution = np.zeros(patterns_length, dtype=int)
        for rod_length, length in ants[curr_min_idx].path:
            solution[length] += 1

        nochange_count = 0
        best_cost = curr_best_cost
        best_solution = solution
        best_used = calc_completion_lenghts(solution, need, patterns)

    # 更新信息素,用平均成本/蚂蚁的成本更新信息素
    for ant in ants:
        for rod_length, choice in ant.path:
            pheromone[rod_length][choice] += avg_cost / ant.cost
    pheromone *= rho

    # 如果数量匹配，且连续100次没有改进，则退出循环
    if np.array_equal(best_used, need) and nochange_count>20:
        print("已达到目标，退出循环")
        break   

    print(f"{iteration}: 最佳成本: {best_cost} 当前平均成本: {curr_avg_cost} 最佳路径: {best_used} 目标: {need} 停滞次数: {nochange_count}")

# 打印最佳解决方案
bar_lengths = np.zeros(len(need),dtype=int)
for i, key in enumerate(patterns):
    bar_lengths += patterns[key][0]*best_solution[i]
        
loss  = np.sum([num*patterns[i][1] for i,num in enumerate(best_solution)])
joint = np.sum([num*patterns[i][2] for i,num in enumerate(best_solution)])
cost  = np.sum([num*patterns[i][3] for i,num in enumerate(best_solution)])

print("最佳方案为：")
# 将最佳方案的组合输出
for i,num in enumerate(best_solution):
    if num > 0:
        print(num, '*', patterns[i][-1])
        
print("最后结果:", bar_lengths, "目标:", need)
print("废料长度:", loss)
print("接头数量:", joint)
print("总成本:", cost)