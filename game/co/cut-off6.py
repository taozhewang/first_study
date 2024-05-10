#%%
import numpy as np
import random

from core import pattern_oringin_by_sampling, calc_cost_by_unmatched, calc_completion_lenghts

'''
用粒子群算法求解钢筋切割问题

目标: [552 658 462] 已完成: [552 477 454] 还差: [  0 181   8]
已有成本: 105751.088 已有损失: 8050 已有接头: 403
还需成本: 39050.28799999999 还需损失: 3050 还需接头: 51
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
need = np.array([552, 658, 462], dtype=int)

# 初始化单个组合的最大数量
max_num = 1
# 最大的组合长度
radius = 14
# 组合的采样数量
sampling_count = 10000
# 最大停滞次数
max_stagnation = 20

# 粒子群算法参数
population_size = 500  # 粒子数量
max_iter = 10000  # 最大迭代次数
c1 = 2  # 个体学习因子
c2 = 2  # 社会学习因子

# 计算适应度函数
def fitness(solution, patterns):
    hascut_lengths = np.zeros_like(need)
    cost = 0
    for i in patterns:
        hascut_lengths += patterns[i][0]*solution[i]
        cost += patterns[i][3]*solution[i]

    # 如果组合的长度不足以切割目标钢筋，这里多匹配和少匹配都算到里面
    bar_lengths = need - hascut_lengths
    # 计算尾料的成本
    cost += calc_cost_by_unmatched(bar_lengths, l, L_values, l_size)[0]
    # 返回成本 
    return cost

# 求各种组合的列表
patterns = pattern_oringin_by_sampling(l, L, sampling_count, radius)
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")

# 初始化种群
population = np.random.randint(0, max_num+1, size=(population_size, patterns_length))
velocities = np.zeros((population_size, patterns_length))
pbest = population.copy()
gbest = np.zeros(patterns_length)

# 初始化个体最优和全局最优
pbest_fitness = np.array([fitness(p, patterns) for p in population])
gbest_fitness = np.min(pbest_fitness)
gbest_index = np.argmin(pbest_fitness)
gbest = population[gbest_index].copy()
nochange_count = 0
# 迭代优化
for i in range(max_iter):
    for j in range(population_size):
        # 更新速度和位置
        velocities[j] = velocities[j] + c1 * random.random() * (pbest[j] - population[j]) + \
                        c2 * random.random() * (gbest - population[j])
                
        velocities[j] = np.clip(velocities[j], -1, 1)  # 限制速度范围
        if np.all(velocities[j] == 0):  # 随机初始化速度
            velocities[j] = np.random.rand(patterns_length)
        
        population[j][velocities[j]<-0.1] -= 1   # 如果速度小于0，则减少1

        population[j][velocities[j]>0.1] += 1   # 如果速度大于0，则增加1

        population[population<0] = 0  # 限制钢筋数量范围

        # 更新个体最优
        sol_fitness = fitness(population[j], patterns)
        if sol_fitness < pbest_fitness[j]:
            pbest[j] = population[j].copy()
            pbest_fitness[j] = sol_fitness
            
        # 更新全局最优
        if sol_fitness < gbest_fitness:
            gbest_fitness = sol_fitness
            gbest = population[j].copy()
            nochange_count = 0

    best_used = calc_completion_lenghts(gbest, need, patterns)        
    print(f"{i}: 平均成本: {np.mean(pbest_fitness)}, 最佳成本: {gbest_fitness}, 最佳完成度: {best_used} 目标: {need} 停滞次数: {nochange_count}/{max_stagnation}")

    nochange_count += 1
    # 如果达到最大停滞次数没有改进，则退出循环
    if nochange_count>max_stagnation:
        print("已达到目标，退出循环")
        break          

# 打印最佳解决方案
bar_lengths = np.zeros(len(need),dtype=int)
for i, key in enumerate(patterns):
    bar_lengths += patterns[key][0]*gbest[i]
        
loss  = np.sum([num*patterns[i][1] for i,num in enumerate(gbest)])
joint = np.sum([num*patterns[i][2] for i,num in enumerate(gbest)])
cost  = np.sum([num*patterns[i][3] for i,num in enumerate(gbest)])

print("最佳方案为：")
# 将最佳方案的组合输出
for i,num in enumerate(gbest):
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