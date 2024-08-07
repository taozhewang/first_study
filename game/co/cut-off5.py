#%%
import numpy as np
import random

from core import pattern_oringin, calc_cost_by_unmatched, calc_completion_lenghts

'''
用蚁群算法求解钢筋切割问题

最佳方案为：
110 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
1 * ['L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
1 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3']
37 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2']
1 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3', 'L3', 'L3']
1 * ['L1', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3']
1 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3']
1 * ['L2', 'L3', 'L3', 'L3', 'L3']
目标: [552 658 462] 已完成: [552 656 462] 还差: [0 2 0]
已有成本: 103112.04800000001 已有损失: 7800 已有接头: 455
还需成本: 41699.328 还需损失: 3300 还需接头: 0
总损失: 11100
总接头: 455
总成本: 144811.37600000002
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

# 最大的组合长度
pattern_radius = 10
# 最大的损失长度
pattern_limit_loss = 1000
# 最大停滞次数
max_stagnation = 100
# 路径系数
path_coef = 2

# 蚁群算法参数
# 最大循环次数
max_iterations = 1000000
# 蚂蚁数量
ant_count = 100  
# 信息素持久因子
rho = 0.2  
# 信息素重要程度因子
alpha = 1 

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
    cost += calc_cost_by_unmatched(bar_lengths, l, L_values, l_size)[0]
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

    # # 计算剩余长度的路径ID, 这里将超过max_pattern_length长度的部分都归为一个路径
    # def cut_off_to_rod_length(self, has_cut_off):

    #     return np.sum([(
            
    #         [i]+1 if has_cut_off[i] < max_pattern_length[i] else 0) * (10**i) for i in range(len(has_cut_off))])
    #     # return np.sum([has_cut_off[i] * (10**i) for i in range(len(has_cut_off))])
    
    def cut_off_to_rod_length(self, has_cut_off):
        k  =1
        id =0
        for i in range(len(has_cut_off)):
            if i>0: k*=max_pattern_length[i-1]
            if has_cut_off[i] < max_pattern_length[i]*path_coef:
                id+=(has_cut_off[i]+1)*k
        return id           
  
        # def idxn(X):
        #     n = len(X)
        #     N = np.zeros(n)
        #     Y = np.ones(n)
        #     for i in range(n):
        #         N[i] = np.sum(X[ : i + 1])
        #         for j in range(i + 1):
        #             Y[i] *= (N[i] + j) / (j + 1)
        #     # print(N, Y)
        #     return np.sum(Y)
        # return int(idxn(has_cut_off))
        # return np.sum([(
            
        #     [i]+1 if has_cut_off[i] < max_pattern_length[i] else 0) * (10**i) for i in range(len(has_cut_off))])

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
                probabilities = pheromone[loa_lengths]**alpha*heuristic + 1e-10
                # probabilities[probabilities==1e-10]=0
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
patterns = pattern_oringin(l, L, pattern_radius, l_min=l_min, l_limit=pattern_limit_loss, only_loss_zero=False)
patterns_length = len(patterns)
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")
patterns_lengths = np.array([patterns[i][0] for i in range(patterns_length)])
max_pattern_length = np.max(patterns_lengths,axis=0)
print(f"max_pattern_length: {max_pattern_length}")
patterns_costs = np.array([patterns[i][3] for i in range(patterns_length)])
# 按成本的倒数计算组合的概率
patterns_c = 1/patterns_costs
patterns_p = patterns_c/np.sum(patterns_c)

# 路径的长度，也就是状态的数量，这里不允许超量切割，所以是钢筋的剩余状态hash数量 
rod_length = np.prod(max_pattern_length*path_coef+1)
# rod_length = np.max(need) ** 3
# rod_length = np.sum([(max_pattern_length[i]+1) * (10**i) for i in range(len(need))])
# rod_length = np.sum([need[i] * (10**i) for i in range(len(need))])
# 初始化信息素矩阵 从一个状态到另外一个状态的概率
pheromone = np.ones((rod_length, patterns_length))

# 初始化启发式信息，这个是个常数，不用更新，表示的是从一个状态到另一个状态的概率
heuristic = patterns_p 

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
            # pheromone[rod_length][choice] += avg_cost / ant.cost
            pheromone[rod_length][choice] += 1 / ant.cost
            # pheromone[rod_length][choice] = pheromone[rod_length][choice]*rho + 1/ant.cost
    pheromone *= rho

    # 如果达到最大停滞次数没有改进，则退出循环
    if nochange_count>max_stagnation:
        print("已达到目标，退出循环")
        break   

    print(f"{iteration}: 当前平均成本: {curr_avg_cost} 最佳成本: {best_cost} 最佳路径: {best_used} 目标: {need} 停滞次数: {nochange_count}/{max_stagnation}")

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

diff = need - bar_lengths
diff_cost, diff_loss, diff_joint = calc_cost_by_unmatched(diff, l, L_values, l_size,l_min)
print(f"目标: {need} 已完成: {bar_lengths} 还差: {diff}")
print(f"已有成本: {cost} 已有损失: {loss} 已有接头: {joint}")
print(f"还需成本: {diff_cost} 还需损失: {diff_loss} 还需接头: {diff_joint}")
print(f"总损失: {loss+diff_loss}")
print(f"总接头: {joint+diff_joint}")
print(f"总成本: {cost+diff_cost}")