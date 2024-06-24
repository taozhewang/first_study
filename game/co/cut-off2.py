#%%
import numpy as np
import random

from core import pattern_oringin, calc_cost_by_unmatched
from core_bak import pattern_oringin4 
'''
用遗传算法求解钢筋切割问题

最佳方案为：
98 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
4 * ['L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3']
1 * ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2']
2 * ['L1', 'L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3']
4 * ['L1', 'L1', 'L1', 'L3', 'L3', 'L3', 'L3', 'L3']
5 * ['L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L3', 'L3']
目标: [552 658 462] 已完成: [540 238 457] 还差: [ 12 420   5]
已有成本: 38731.248 已有损失: 2800 已有接头: 335
还需成本: 106070.128 还需损失: 8300 还需接头: 119
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
# 变异个数为1
variation_count = 2

# 遗传算法参数
pop_size = 200  # 种群大小
gen_max = 1000  # 进化次数
mut_rate = 1  # 变异率

# 计算当前组合最终完成的长度
# individual: [0,...,] 表示选择该种组合的选择索引,长度为 dna_size 的长度， -1 表示无效
# patterns: 所有组合的列表，每个元素为 [counter, loss, joint, cost, eer, combin]
def calc_hascut_lenghts(individual, patterns):
    hascut_lengths = np.zeros_like(need)
    for i in range(len(individual)):
        hascut_lengths += patterns[i][0]*individual[i]
    return hascut_lengths

# 适应度函数，这里按成本来计算, 越低越好
# individual: [[0,...,],pop_size] 表示选择该种组合的选择索引,长度为 dna_size 的长度， -1 表示无效
# patterns: 所有组合的列表，每个元素为 [counter, loss, joint, cost, eer, combin]
def fitness(population, patterns_costs, patterns_lengths):
    cost = population.dot(patterns_costs)
    hascut_lengths = population.dot(patterns_lengths)    
    for i in range(len(population)):
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

patterns_costs = np.array([patterns[i][3] for i in range(patterns_length)])
print(patterns_costs)
patterns_lengths = np.array([patterns[i][0] for i in range(patterns_length)])

# 按成本的倒数计算组合的概率
patterns_p = 1/patterns_costs
patterns_p = patterns_p/np.sum(patterns_p)
dna_size = patterns_length  # DNA长度

# 初始化种群 
# dna 保存 patterns 数量， 
population = np.zeros((pop_size, dna_size),dtype=int)

# 记录最佳适应度个体
best_individual=None
# 记录最佳适应度
best_fitnesses=np.inf
# 记录最佳耗用
best_used = None

# 最大停滞次数
max_stagnation = 200
# 进化停顿次数
nochange_count = 0

# 进化循环
for gen in range(gen_max):
    # 评估适应度
    fitnesses = fitness(population, patterns_costs, patterns_lengths)
    min_idx = np.argmin(fitnesses)
    max_idx = np.argmax(fitnesses)
                        
    # 记录最佳适应度个体
    if best_fitnesses > fitnesses[min_idx]:
        best_fitnesses = fitnesses[min_idx]
        best_individual = np.copy(population[min_idx])
        best_used = calc_hascut_lenghts(best_individual, patterns)
        # 如果最佳个体发生变化，将计数器清零
        nochange_count = 0
            
    nochange_count += 1
    
    # 选择一半最小适应度的个体作为父代
    parents = np.array([population[i] for i in np.argsort(fitnesses)[:pop_size//2]])  
       
    # 交叉
    offspring = []
    while len(offspring)<pop_size:    
        parent1, parent2 = random.sample(list(parents), 2) # 随机抽取父类        
        crossover_point = random.randint(1, patterns_length-1)
        offspring.append(np.concatenate((parent1[:crossover_point], parent2[crossover_point:])))  
        offspring.append(np.concatenate((parent2[:crossover_point], parent1[crossover_point:])))
        
    # 变异 按概率变异
    for i in range(len(offspring)):
        if random.random() < mut_rate:
            ids = np.random.choice(patterns_length, variation_count, replace=False, p=patterns_p)
            for id in ids:                
                if random.random()<0.5:
                    if offspring[i][id]>0:
                        offspring[i][id] -= 1
                else:
                    offspring[i][id] += 1

        # 限制最大长度
        while np.any(need-calc_hascut_lenghts(offspring[i], patterns)<0): 
            while True:
                id = np.random.choice(patterns_length)
                if offspring[i][id]>0:
                    offspring[i][id] -= 1
                    break
    # 替换为新种群
    population = np.array(offspring)
    

    print(f"{gen}: 平均成本：{np.mean(fitnesses)} 最低成本：{best_fitnesses} 完成度: {best_used} 目标: {need} 停滞次数: {nochange_count}/{max_stagnation}")
    
    # 如果达到最大停滞次数没有改进，则退出循环
    if nochange_count>max_stagnation:
        print("已完成目标，停止进化")
        break

# 打印最佳解决方案
bar_lengths = best_individual.dot(patterns_lengths)
        
loss   = np.sum([patterns[i][1]*num for i,num in enumerate(best_individual)])
joint  = np.sum([patterns[i][2]*num for i,num in enumerate(best_individual)])
cost   = np.sum([patterns[i][3]*num for i,num in enumerate(best_individual)])


print("最佳方案为：")            
for key,num in enumerate(best_individual):
    if num>0:
        print(num, '*', patterns[key][-1])

diff = need - bar_lengths
diff_cost, diff_loss, diff_joint = calc_cost_by_unmatched(diff, l, L_values, l_size,l_min)
print(f"目标: {need} 已完成: {bar_lengths} 还差: {diff}")
print(f"已有成本: {cost} 已有损失: {loss} 已有接头: {joint}")
print(f"还需成本: {diff_cost} 还需损失: {diff_loss} 还需接头: {diff_joint}")
print(f"总损失: {loss+diff_loss}")
print(f"总接头: {joint+diff_joint}")
print(f"总成本: {cost+diff_cost}")
