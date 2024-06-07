#%%
import numpy as np
import random

from core import pattern_oringin, calc_cost_by_unmatched, calc_completion_lenghts

'''
用禁忌搜索算法求解钢筋切割问题

最佳方案为：
87 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3'] 0 3
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3'] 0 4
5 * ['L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3'] 50 2
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3'] 50 4
3 * ['L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3'] 100 3
1 * ['L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2'] 100 4
7 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3'] 150 2
1 * ['L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2'] 150 3
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3'] 150 4
4 * ['L1', 'L1', 'L1', 'L3', 'L3', 'L3', 'L3', 'L3'] 200 2
2 * ['L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3'] 200 3
2 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L3', 'L3', 'L3', 'L3'] 200 4
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3'] 200 5
2 * ['L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3'] 300 2
1 * ['L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L3'] 300 3
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3'] 350 3
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L3', 'L3'] 350 5
1 * ['L1', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3'] 400 2
目标: [552 658 462] 已完成: [548 313 454] 还差: [  4 345   8]
已有成本: 77491.536 已有损失: 5850 已有接头: 357
还需成本: 67309.84 还需损失: 5250 还需接头: 97
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

# 禁忌搜索参数
# 最大循环次数
max_iterations = 1000000
# 禁忌表大小
tabu_tenure = 500
# 变异个数
variation_count = 3
# 最大停滞次数
max_stagnation = 1000

# 初始化解
# patterns_length 组合的长度
# max_num 最大的组合数量
def init_solution(patterns_length):
    return np.zeros(patterns_length, dtype=int)

# 评估函数
def evaluate(solutions, need, patterns_lengths, patterns_costs):
    cost = solutions.dot(patterns_costs)
    hascut_lengths = solutions.dot(patterns_lengths)    
    for i in range(len(cost)):
        # 如果组合的长度不足以切割目标钢筋，这里多匹配和少匹配都算到里面
        bar_lengths = need - hascut_lengths[i]
        # 计算尾料的成本
        cost[i] += calc_cost_by_unmatched(bar_lengths, l, L_values, l_size)[0]    
    return cost

# 求各种组合的列表
patterns = pattern_oringin(l, L, pattern_radius, l_min=l_min, l_limit=pattern_limit_loss, only_loss_zero=False)
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")
patterns_lengths = np.array([patterns[i][0] for i in range(patterns_length)])
patterns_costs = np.array([patterns[i][3] for i in range(patterns_length)])
# 按成本的倒数计算组合的概率
patterns_p = 1/patterns_costs
patterns_p = patterns_p/np.sum(patterns_p)

# 邻域操作
def get_neighbor(solution, patterns_length, variation_count, patterns_p):    
    neighbor = np.copy(solution)

    ids = np.random.choice(patterns_length, variation_count, replace=False, p=patterns_p)
    for idx in ids:
        v = 1 if random.random()<0.5 else -1
        neighbor[idx] += v
    neighbor[neighbor<0] = 0

    return neighbor

# 禁忌搜索,检查邻域解是否在禁忌表中
def check_tabu(tabu_list, neighbor):
    for tabu in tabu_list:
        if np.array_equal(tabu, neighbor):
            return True
    return False

# 禁忌搜索算法
def tabu_search(max_iterations, tabu_tenure, patterns_length, variation_count, patterns_p):
    # 采用随机初始解
    tabu_list = np.array([init_solution(patterns_length) for i in range(tabu_tenure)])
    tabu_waste_list = evaluate(tabu_list, need, patterns_lengths, patterns_costs)

    # 记录最佳解
    best_solution = None
    # 记录最佳解的评估
    best_waste = np.inf
    # 记录连续没有改进的次数
    nochange_count = 0

    for i in range(max_iterations):
        # 从禁忌表中获得一组邻域解
        neighbors = np.array([get_neighbor(solution, patterns_length, variation_count, patterns_p) for solution in tabu_list])
        # 计算邻域解的评估
        neighbors_waste = evaluate(neighbors, need, patterns_lengths, patterns_costs)
        # 选择最佳邻域解
        best_idx = np.argmin(neighbors_waste)
        best_neighbor, best_neighbor_waste = neighbors[best_idx], neighbors_waste[best_idx]       

        # 禁忌搜索
        # 如果邻域解比最佳解好，更新最佳解
        if best_neighbor_waste < best_waste:
            best_solution = best_neighbor
            best_waste = best_neighbor_waste
            nochange_count = 0
            
        nochange_count += 1

        # 如果邻域解比当前解好，且邻域解不在禁忌表中，则更新禁忌组
        update_count = 0
        avg_waste = sum(tabu_waste_list)/len(tabu_waste_list)
        for idx, waste in enumerate(neighbors_waste):
            if waste <= avg_waste and not check_tabu(tabu_list, neighbors[idx]):
            # if waste < tabu_waste_list[idx] and not check_tabu(tabu_list, neighbors[idx]):
                # 记录最佳解
                update_count += 1
                worst_idx = np.argmax(tabu_waste_list)
                tabu_list[worst_idx]=neighbors[idx]
                tabu_waste_list[worst_idx]=waste

        if i % 10 == 0:
            best_used = calc_completion_lenghts(best_solution, need, patterns)

            print(f"{i}: 禁忌组平均成本:{avg_waste}, 更新个数:{update_count}, 最佳成本:{best_waste}, 最佳完成度: {best_used} 目标: {need} 停滞次数: {nochange_count}/{max_stagnation}")

            # 如果达到最大停滞次数没有改进，则退出循环
            if  nochange_count>max_stagnation:
                print("已达到目标，退出循环")
                break            

    return best_solution, best_waste

best_solution, best_waste = tabu_search(max_iterations, tabu_tenure, patterns_length, variation_count, patterns_p)

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
        print(num, '*', patterns[i][-1], patterns[i][1], patterns[i][2])

diff = need - bar_lengths
diff_cost, diff_loss, diff_joint = calc_cost_by_unmatched(diff, l, L_values, l_size,l_min)
print(f"目标: {need} 已完成: {bar_lengths} 还差: {diff}")
print(f"已有成本: {cost} 已有损失: {loss} 已有接头: {joint}")
print(f"还需成本: {diff_cost} 还需损失: {diff_loss} 还需接头: {diff_joint}")
print(f"总损失: {loss+diff_loss}")
print(f"总接头: {joint+diff_joint}")
print(f"总成本: {cost+diff_cost}")
