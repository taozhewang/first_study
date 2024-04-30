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
max_num = 2 #np.max(need)

# 最大的组合长度
radius = 10
# 组合数最小余料
losses1 = 30

# 遗传算法参数
pop_size = 200  # 种群大小
gen_max = 1000  # 进化次数
mut_rate = 0.5  # 变异率

# 适应度函数，这里按成本来计算, 越低越好
# individual: [0,...,] 表示选择该种组合的选择数量,长度为patterns的长度
# patterns: 所有组合的列表，每个元素为 [counter, loss, joint, cost, eer, combin]
def calc_hascut_lenghts(individual, patterns):
    hascut_lengths = np.zeros_like(need)
    for i in patterns:
        hascut_lengths += patterns[i][0]*individual[i]
    return hascut_lengths

def fitness(individual, patterns):
    hascut_lengths = np.zeros_like(need)
    cost = 0
    for i in patterns:
        hascut_lengths += patterns[i][0]*individual[i]
        cost += patterns[i][3]*individual[i]

    # 如果组合的长度不足以切割目标钢筋，这里多匹配和少匹配都算到里面
    bar_lengths = need - hascut_lengths
    # 计算尾料的成本,这里不鼓励没有分配完，所以*10倍的系数作为惩罚
    dl=np.array(list(L.values()))
    loss, joint = calc_loss_joint(bar_lengths, l, dl, l_min)
    cost += calc_cost(loss, joint, l_size)
    return cost, bar_lengths


# 求各种组合的列表
patterns = pattern_oringin(l, L, losses1, radius)
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")# 产生patterns，最低1个组合，因为需要处理尾料

# 初始化种群 
population = np.random.randint(0, max_num, (pop_size, patterns_length))  # 种群初始化，二进制编码

# 记录最佳适应度个体
best_individual=None
# 记录最佳适应度
best_fitnesses=np.inf
# 初始化变异个数为整体的1/8
change_number = patterns_length//8

# 进化停顿次数
nochange_count = 0
# 进化停止条件
stop_count = 0

# 进化循环
for gen in range(gen_max):
    # 评估适应度
    fitnesses = np.zeros(pop_size)
    for i in range(pop_size):
        individual = population[i]
        cost, number = fitness(individual, patterns)

        # 适应度包含了成本和完成距离目标的距离的平方，让算法更注重完成距离
        fitnesses[i] = cost+np.sum(np.power(number,2))

        # 记录最佳适应度个体
        if best_fitnesses > fitnesses[i]:
            best_fitnesses = fitnesses[i]
            best_individual = individual
            # 如果最佳个体发生变化，将计数器清零
            nochange_count = 0
            stop_count = 0
            # 计算需要变异的数量,前期多变异，后期少变异，最低保留2处变异地方
            change_number = np.sum(np.abs(number))
            change_number = change_number//10
            if change_number<2:
                change_number=2

    nochange_count += 1
    
    # 进化停止条件，如果十轮都没有进化，则停止
    if stop_count >= 10:
        print("进化停止，终止")
        break

    # 选择一半最小适应度的个体作为父代
    parents = np.array([population[i] for i in np.argsort(fitnesses)[:pop_size//2]])  
    if nochange_count >= 10:
        print("进化停止，强制复制最佳个体到父代，增加成功率")
        parents = np.concatenate((parents,[best_individual for i in range(pop_size//2)]))
        stop_count += 1 
        nochange_count = 0
        # 同时增加变异的个数，增加变异成功率
        change_number += 1
       
    # 交叉
    offspring = []
    for i in range(pop_size//2):    
        parent1, parent2 = random.sample(list(parents), 2) # 随机抽取父类        
        crossover_point = random.randint(1, patterns_length-1)
        offspring.append(np.concatenate((parent1[:crossover_point], parent2[crossover_point:])))
        offspring.append(np.concatenate((parent2[:crossover_point], parent1[crossover_point:])))

    # 变异 按概率变异
    for i in range(len(offspring)):
        if random.random() < mut_rate:
            # 随机选择变异的方式，0 数量变成 0， 1 数量变成+1 
            # 为了防止变异互相抵消，这里变异采用统一方向
            v = 0 if random.random()>0.5 else 1            
            for _ in range(change_number):
                if change_number==1:
                    idx=random.randint(0, 10)
                else:
                    idx = random.randint(0, patterns_length-1)

                if v == 0:
                    offspring[i][idx] = 0
                else:
                    offspring[i][idx] += 1 

    # 替换为新种群
    population = np.array(offspring)

    print(f"进化次数：{gen}, 最佳适应度(成本)：{np.min(fitnesses)}, 平均适应度(成本)：{np.mean(fitnesses)},\
        最佳: {calc_hascut_lenghts(best_individual, patterns)} 目标: {need} 变异个数: {change_number}")


# 打印最佳解决方案
bar_lengths = np.zeros(len(need),dtype=int)
for i, key in enumerate(patterns):
    bar_lengths += patterns[key][0]*best_individual[i]
        
loss  = np.sum([num*patterns[i][1] for i,num in enumerate(best_individual)])
joint = np.sum([num*patterns[i][2] for i,num in enumerate(best_individual)])
cost  = np.sum([num*patterns[i][3] for i,num in enumerate(best_individual)])

print("最佳方案为：")
# 将最佳方案的组合输出
for i,num in enumerate(best_individual):
    if num > 0:
        print(num, '*', patterns[i][-1])
        
print("最后结果:", bar_lengths, "目标:", need)
print("废料长度:", loss)
print("接头数量:", joint)
print("总成本:", cost)