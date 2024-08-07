#%%
import numpy as np
import random

from core import pattern_oringin, calc_cost_by_unmatched, calc_completion_lenghts

'''
用模拟退火算法求解钢筋切割问题

最佳方案为：
51 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
5 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3']
2 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2']
23 * ['L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
4 * ['L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3']
8 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3']
7 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2']
1 * ['L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3']
4 * ['L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2']
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3']
27 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3']
7 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2']
4 * ['L1', 'L1', 'L1', 'L3', 'L3', 'L3', 'L3', 'L3']
3 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L3', 'L3', 'L3', 'L3']
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L3', 'L3', 'L3']
1 * ['L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
2 * ['L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3']
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3']
目标: [552 658 462] 已完成: [540 634 459] 还差: [12 24  3]
已有成本: 140900.528 已有损失: 10800 已有接头: 443
还需成本: 3900.848 还需损失: 300 还需接头: 11
总损失: 11100
总接头: 454
总成本: 144801.376
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
need = np.array([552, 658, 462],dtype=int)

# 最大的组合长度
pattern_radius = 10
# 最大的损失长度
pattern_limit_loss = 500
# 最小变异个数
variation_count = 2

# 模拟退火参数
# 最大循环次数
max_iterations = 1000000
# 最大温度
max_temperature = 100
# 退火速率
cooling_rate = 0.99
# 最大停滞次数
max_stagnation = 100000

# 初始化解
# patterns_length 组合的长度
# max_num 最大的组合数量
def init_solution(patterns_length):
    return np.zeros(patterns_length,dtype=int)

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

# 求各种组合的列表
patterns = pattern_oringin(l, L, pattern_radius, l_min=l_min, l_limit=pattern_limit_loss, only_loss_zero=False)
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")
patterns_costs = np.array([patterns[i][3] for i in range(patterns_length)])
# 按成本的倒数计算组合的概率
patterns_p = 1/patterns_costs
patterns_p = patterns_p/np.sum(patterns_p)

# 邻域操作
def get_neighbor(solution, patterns_length, variation_count):
    neighbor = np.copy(solution)
    ids = np.random.choice(patterns_length, variation_count, replace=False, p=patterns_p)
    for idx in ids:
        neighbor[idx] += 1 if random.random()<0.5 else -1
        if neighbor[idx] < 0: neighbor[idx]= 0
    return neighbor

# 模拟退火算法
def simulated_annealing(max_iterations, max_temperature, cooling_rate, variation_count):
    # 当前温度
    temperature = max_temperature
    # 当前解
    current_solution = init_solution(patterns_length)
    # 计算当前解的评估值
    current_waste = evaluate(current_solution, need, patterns)
    # 最佳解
    best_solution = np.copy(current_solution)
    # 最佳解的评估值
    best_waste = current_waste
    # 连续无改进次数
    nochange_count = 0

    # 动态调整异动个数
    for i in range(max_iterations):

        if i%100 == 0:
            best_used = calc_completion_lenghts(best_solution, need, patterns)

            print(f"{i}: 当前成本={current_waste} 最佳成本={best_waste} 最佳完成度: {best_used} 目标: {need} 温度: {temperature} 停滞次数: {nochange_count}/{max_stagnation}")
            # 如果达到最大停滞次数没有改进，则退出循环
            if  nochange_count>max_stagnation:
                print("已达到目标，退出循环")
                break

        # 如果长期没有减,温度降低一下
        if nochange_count%100 == 0:
            temperature *= cooling_rate

        nochange_count += 1
        # 产生邻域解
        neighbor = get_neighbor(current_solution, patterns_length, variation_count)
        # 计算邻域解的评估值
        neighbor_waste = evaluate(neighbor, need, patterns)

        # 计算差距
        delta = (neighbor_waste - current_waste)
        if delta <= 0:
            # 如果成本减少，接受邻域解
            current_solution = neighbor
            current_waste = neighbor_waste
            if neighbor_waste < best_waste:
                best_solution = np.copy(neighbor)
                best_waste = neighbor_waste
                nochange_count = 0
        else:
            # 如果成本增加，概率接收邻域解
            probability = np.exp(-delta / temperature)
            if random.uniform(0, 1) < probability:
                # print("接受邻域解", probability, delta, temperature)
                current_solution = neighbor
                current_waste = neighbor_waste

    return best_solution, best_waste

best_solution, best_waste = simulated_annealing(max_iterations, max_temperature, cooling_rate, variation_count)

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

