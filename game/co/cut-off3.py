#%%
import numpy as np
import random

from core import pattern_oringin_by_sampling, calc_cost_by_unmatched, calc_completion_lenghts

'''
用模拟退火算法求解钢筋切割问题

目标: [552 658 462] 已完成: [540 216 432] 还差: [ 12 442  30]
已有成本: 3240.0 已有损失: 0 已有接头: 324
还需成本: 141611.37600000002 还需损失: 11100 还需接头: 135
总损失: 11100
总接头: 459
总成本: 144851.37600000002
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

# 初始化单个组合的最大数量
max_num = 3
# 最大的组合长度
radius = 13
# 组合的采样数量
sampling_count = 5000
# 最小变异个数
min_variation_count = 3

# 模拟退火参数
# 最大循环次数
max_iterations = 1000000
# 最大温度
max_temperature = 100000
# 退火速率
cooling_rate = 0.99
# 最大停滞次数
max_stagnation = 10000

# 初始化解
# patterns_length 组合的长度
# max_num 最大的组合数量
def init_solution(patterns_length, max_num):
    return np.random.randint(0, max_num+1, patterns_length)   

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
patterns = pattern_oringin_by_sampling(l, L, sampling_count, radius)
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")

# 邻域操作
def get_neighbor(solution, patterns_length, variation_count):
    neighbor = np.copy(solution)
    ids = np.random.choice(patterns_length, variation_count, replace=False)
    # 为了加快收敛，变异方向固定
    v = 1 if random.random()<0.5 else -1
    for idx in ids:
        # 如果只剩下2个变异位置，则变异方向随机
        if variation_count==min_variation_count: v = 1 if random.random()<0.5 else -1
        neighbor[idx] += v
        if neighbor[idx] < 0: neighbor[idx]= 0
    return neighbor

# 模拟退火算法
def simulated_annealing(max_iterations, max_temperature, cooling_rate):
    # 当前温度
    temperature = max_temperature
    # 当前解
    current_solution = init_solution(patterns_length, max_num)
    # 计算当前解的评估值
    current_waste = evaluate(current_solution, need, patterns)
    # 最佳解
    best_solution = np.copy(current_solution)
    # 最佳解的评估值
    best_waste = current_waste
    # 连续无改进次数
    nochange_count = 0

    # 动态调整异动个数
    variation_count = patterns_length//8    
    for i in range(max_iterations):

        if i%100 == 0:
            best_used = calc_completion_lenghts(best_solution, need, patterns)
            # 动态调整异动个数
            variation_count = np.sum(np.abs(best_used-need))//20
            if variation_count>patterns_length//2: variation_count=patterns_length//2
            if variation_count<min_variation_count: variation_count=min_variation_count

            print(f"{i}: 当前成本={current_waste} 最佳成本={best_waste} 最佳完成度: {best_used} 目标: {need} 异动个数: {variation_count} 温度: {temperature} 停滞次数: {nochange_count}/{max_stagnation}")
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

best_solution, best_waste = simulated_annealing(max_iterations, max_temperature, cooling_rate)

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

