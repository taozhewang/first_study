#%%
import numpy as np
import random

from core import pattern_oringin, calc_loss_joint, calc_cost

'''
用遗传算法求解钢筋切割问题
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
need = np.array([552, 658, 462],dtype=int)
max_num = np.max(2)

# 最大的组合长度
radius = 10
# 组合数最小余料
losses1 = 30

# 遗传算法参数
pop_size = 1000  # 种群大小
gen_max = 200  # 进化次数
mut_rate = 0.1  # 变异率


# 适应度函数，这里按成本来计算, 越低越好
# individual: 0-pop_size 之间的数组，表示选择该种组合的选择数量
# patterns: 所有组合的列表，每个元素为 [{key1:count,key2:count,...}, loss, joint, cost, eer]
def fitness(individual, patterns):
    bar_lengths = np.zeros(len(need),dtype=int)
    cost = 0
    for i in range(len(patterns)):
        bar_lengths += np.array([patterns[i][0][key]*individual[i] for key in patterns[i][0]])
        cost += patterns[i][3]*individual[i]

    # 如果组合的长度不足以切割目标钢筋，则单独计算尾料的成本
    bar_lengths = need - bar_lengths
    dl=list(L.values())
    loss, joint = calc_loss_joint(bar_lengths, l, dl, l_min)
    cost += calc_cost(loss, joint, l_size)            
    return cost


# 求各种组合的列表
patterns = pattern_oringin(l, L, losses1, radius)
'''patterns: {0: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 0,0,0,0],
            1: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 50,3,400,100]}'''
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")# 产生patterns，最低1个组合，因为需要处理尾料

# 初始化种群 
population = np.random.randint(0, max_num, (pop_size, patterns_length))  # 种群初始化，二进制编码

# 进化循环
for gen in range(gen_max):
    # 评估适应度
    fitnesses = np.array([fitness(individual, patterns) for individual in population])
    
    # 选择一半最小适应度的个体作为父代
    parents = np.array([population[i] for i in np.argsort(fitnesses)[:pop_size//2]])  
    
    # 交叉
    offspring = []
    for i in range(pop_size//2):
        parent1, parent2 = random.sample(list(parents), 2)
        crossover_point = random.randint(1, patterns_length-1)
        offspring.append(np.concatenate((parent1[:crossover_point], parent2[crossover_point:])))
        offspring.append(np.concatenate((parent2[:crossover_point], parent1[crossover_point:])))
    
    # 变异
    for i in range(len(offspring)):
        if random.random() < mut_rate:
            idx = random.randint(0, patterns_length-1)
            v = offspring[i][idx]
            if v == 0:
                offspring[i][idx] = 1
            else:
                offspring[i][idx] += 1 if random.random() < 0.5 else -1

    # 替换为新种群
    population = np.array(offspring)

    print(f"进化次数：{gen}, 最佳适应度：{np.min(fitnesses)}, 平均适应度：{np.mean(fitnesses)}")


# 打印最佳解决方案
best_individual = population[np.argmax([fitness(individual, patterns) for individual in population])]
print("最佳解决方案:", best_individual)
print("适应度:", fitness(best_individual, patterns))


