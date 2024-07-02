#%%
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import time


'''
用遗传算法求解钢筋切割问题

最佳方案为：
40 * [1, 3, 4] loss: 50 : [4700, 4350, 4350, 4700, 4350, 4700, 4700, 4100]
25 * [5, 2, 4] loss: 0 : [4100,

 4700, 4100, 4700, 4350, 4350, 4100, 4700, 4700, 4100, 4100]
19 * [5, 8, 1] loss: 0 : [4100, 4350, 4350, 4100, 4350, 4100, 4700, 4350, 4100, 4350, 4350, 4350, 4100, 4350]
33 * [4, 4, 3] loss: 100 : [4100, 4100, 4350, 4100, 4350, 4700, 4350, 4100, 4700, 4700, 4350]
4 * [1, 9, 1] loss: 50 : [4350, 4100, 4350, 4350, 4350, 4350, 4350, 4350, 4700, 4350, 4350]
14 * [0, 5, 3] loss: 150 : [4350, 4350, 4350, 4700, 4700, 4700, 4350, 4350]
11 * [7, 5, 2] loss: 150 : [4700, 4100, 4100, 4700, 4100, 4350, 4350, 4100, 4100, 4350, 4350, 4350, 4100, 4100]
3 * [8, 3, 3] loss: 50 : [4100, 4700, 4350, 4100, 4100, 4700, 4700, 4350, 4100, 4100, 4100, 4100, 4100, 4350]
1 * [12, 2, 3] loss: 0 : [4100, 4100, 4100, 4100, 4700, 4100, 4100, 4100, 4100, 4100, 4700, 4100, 4350, 4100, 4350, 4100, 4700]
1 * [8, 9, 0] loss: 50 : [4100, 4350, 4100, 4350, 4350, 4350, 4350, 4350, 4100, 4100, 4350, 4100, 4350, 4350, 4100, 4100, 4100]
1 * [11, 4, 2] loss: 100 : [4350, 4350, 4100, 4100, 4100, 4100, 4700, 4700, 4350, 4100, 4100, 4100, 4350, 4100, 4100, 4100, 4100]
2 * [12, 8, 0] loss: 0 : [4350, 4350, 4100, 4350, 4100, 4100, 4350, 4100, 4350, 4350, 4100, 4100, 4100, 4350, 4100, 4100, 4350, 4100, 4100, 4100]
1 * [0, 3, 2] loss: 1550 : [4350, 4700, 4350, 4700, 4350]e
总损失: 11100.0
总接头: 454
总成本: 144801.376
用时: 104.87966299057007 秒
'''

# 原始钢筋长度
l = 12000
# 最小可接长度，小于200的部分会做为废料
l_min = 200
# 钢筋的规格
l_size = 32
# 目标钢筋长度
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
L_values = list(L.values())
# 目标钢筋的数量
need = np.array([552, 658, 462],dtype=int)


# 遗传算法参数
pop_size = 100  # 种群大小
gen_max = 100000  # 进化次数
dna_length = np.sum(need)   # 基因长度

# 计算适应度
def fitness(population, l, l_min, l_size):
    combinations_size = len(population)                             # 种群大小
    _l = np.ones(combinations_size)*l                               # 剩余长度
    group_count = np.zeros(combinations_size,dtype=int)             # 小组个数
    group_firstpos = np.zeros(combinations_size,dtype=int)          # 第一个小组的位置
    group_endpos = np.zeros(combinations_size,dtype=int)            # 最后一个小组的位置
    loss = np.zeros(combinations_size,dtype=float)                  # 余料
    joint = np.zeros(combinations_size,dtype=int)                   # 接头
    cost = np.zeros(combinations_size,dtype=float)                  # 成本

    for i in range(len(population[0])): 
        while True:
            idxs=np.where(_l<population[:,i])[0]
            if len(idxs)==0: break
            _l[idxs] += l
            joint[idxs] += 1

        _l -= population[:,i]
        
        # 是否存在左边接头长度不够的情况
        min_idx = np.where((l-_l)<l_min)[0]
        if len(min_idx)>0:
            add_len = l_min - (l-_l)
            loss[min_idx] += add_len[min_idx]
            _l[min_idx] -= add_len[min_idx]

        # 确定第一个小组的最后一个位置,如果第一个位置为0且有废料，则将其作为第一个位置
        fidx = np.where((group_firstpos==0) & (_l<l_min))[0]
        if len(fidx)>0:
            group_firstpos[fidx]= i+1

        idxs=np.where(_l<l_min)[0]
        if len(idxs)>0:
            loss[idxs] += _l[idxs]
            group_count[idxs] += 1
            group_endpos[idxs] = i+1
            _l[idxs] = l

    loss += _l

    cost_param = 0.00617*2000*(l_size**2)/1000
    cost = loss*cost_param + joint*10
    return loss, joint, cost, group_count, group_firstpos, group_endpos

# 找到最长匹配的段
def find_longest_matching_segment(list1, list2, need, keys):    
    to = np.ones(len(keys),dtype=int)
    for i,n in enumerate(need):
        to[i]=n
    for i in range(len(list1)//2,1,-1):
        counter1 = Counter(list1[:i])
        counter2 = Counter(list2[:i])
        c1 = np.zeros(len(keys),dtype=int)
        c2 = np.zeros(len(keys),dtype=int)
        for j, k in enumerate(keys):
            if k in counter1:
                c1[j] = counter1[k]
            if k in counter2:
                c2[j] = counter2[k]
        if np.array_equal(c1,c2):
            return i, 1
        if np.array_equal(c1,to-c2):
            return i, -1
    return -1, -1

# 比较两个group，返回接头数量选择较小的group
def get_best_solution_group(group1, group2, l, l_min):
    joint1 = joint2 = 0

    _l = l
    for length in group1:
        while _l < length:
            _l += l
            joint1 += 1

        _l -= length

        if l-_l<l_min: _l -= l_min-(l-_l)
        if _l < l_min: _l = l

    _l = l
    for length in group2:
        while _l < length:
            _l += l
            joint2 += 1
        _l -= length
        if l-_l<l_min:  _l -= l_min-(l-_l)        
        if _l < l_min:  _l = l

    return group1 if joint1 < joint2 else group2

# 初始化种群 
# dna 保存各个钢筋的长度
_population = []
for i, num in enumerate(need):
    if num>0:
        _population += [L_values[i]]*num
population = np.ones((pop_size, dna_length), dtype=int)
for i in range(pop_size):
    population[i] = np.random.permutation(_population)

# 记录最佳适应度个体
best_individual=None
# 记录最佳适应度
best_fitnesses=np.inf
best_loss=0
best_joint=0

# 最大停滞次数
max_stagnation = 500
# 进化停顿次数
nochange_count = 0


gen_values = []  # 记录每代的最低适应度
gen_times = []   # 记录每代的时间

start_time = time.time()
# 进化循环
for gen in range(gen_max):
    # 评估适应度

    loss, joint, cost, group_count, group_firstpos, group_endpos = fitness(population, l, l_min, l_size)

    # 记录最佳适应度个体
    best_idx = np.argmin(cost)
    best_population_fitnesses = cost[best_idx] 
    if best_population_fitnesses < best_fitnesses:
        best_individual = np.copy(population[best_idx])
        best_fitnesses = best_population_fitnesses
        nochange_count = 0
        best_loss = loss[best_idx]
        best_joint = joint[best_idx]
            
    gen_values.append(best_fitnesses)
    gen_times.append(time.time()-start_time)
    
    nochange_count += 1
    
    # 选择一半最小适应度的个体作为父代
    argsort_cost = np.argsort(cost+1./group_count)
    parents = np.array([population[i] for i in argsort_cost[:pop_size//2]])  

    # 变异
    for i in range(len(parents)):
        first_pos, end_pos = group_firstpos[argsort_cost[i]], group_endpos[argsort_cost[i]]
        if end_pos < dna_length-1:   
            first_group = get_best_solution_group(parents[i][:first_pos], np.random.permutation(parents[i][:first_pos]), l, l_min)
            parents[i] = np.concatenate((parents[i][first_pos:end_pos], first_group, np.random.permutation(parents[i][end_pos:])))
        else:
            parents[i] = np.concatenate((parents[i][first_pos:], np.random.permutation(parents[i][:first_pos])))

    # 交叉
    offspring = []
    for i in range(pop_size//2):    
        while True:
            parent1, parent2 = random.sample(list(parents), 2) # 随机抽取父类    
            crossover_point, flag = find_longest_matching_segment(parent1, parent2, need, L_values) 
            if crossover_point>0: # 找到匹配的点，如果没有，则重新抽取
                if flag==1:
                    offspring.append(np.concatenate((parent1[:crossover_point], parent2[crossover_point:])))
                    offspring.append(np.concatenate((parent2[:crossover_point], parent1[crossover_point:])))
                else:
                    offspring.append(np.concatenate((parent1[:crossover_point], parent2[:crossover_point])))
                    offspring.append(np.concatenate((parent2[crossover_point:], parent1[crossover_point:])))
                break
              
    # 替换为新种群
    population = np.array(offspring)

    print(f"{gen}: 平均成本：{np.mean(cost)} 最低成本：{best_fitnesses} 余料: {best_loss} 接头: {best_joint}  目标: {need} 停滞次数: {nochange_count}/{max_stagnation}")
    
    # 如果达到最大停滞次数没有改进，则退出循环
    if nochange_count>max_stagnation:
        print("已完成目标，停止进化")
        break

# 打印最佳解决方案
# 将排序转为group key:[]
counters=[]
groups=[]
nums=[]
loss=[]
_group=[]
_l=l
names=list(L.keys())
for p in best_individual:
    _group.append(p)
    while _l<p:
        _l+=l
    _l -= p

    if _l<l_min:
        _counter=[0 for _ in range(len(L))]
        for x in _group:
            _counter[L_values.index(x)]+=1
        if _counter in counters:
            nums[counters.index(_counter)]+=1
        else:    
            counters.append(_counter)
            groups.append(_group.copy())
            nums.append(1)
            loss.append(_l)
        _group=[]
        _l=l
   
if len(_group)>0:
    _counter=[0 for _ in range(len(L))]
    for x in _group:
        _counter[L_values.index(x)]+=1
    counters.append(_counter)
    groups.append(_group.copy())
    nums.append(1)
    loss.append(_l)
    
print("最佳方案为：")
for i in range(len(nums)):
    print(nums[i],'*',counters[i],"loss:",loss[i],":",groups[i])
           
# print(need, Counter(best_individual))
print(f"总损失: {best_loss}")
print(f"总接头: {best_joint}")
print(f"总成本: {best_fitnesses}")
print(f"用时: {time.time() - start_time} 秒")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.plot(gen_times, gen_values, marker='o', label='最低成本')
plt.xlabel('时间(s)')
plt.ylabel('目标函数值')
plt.title('遗传搜索算法收敛速度图')
plt.grid(True)
plt.legend()
plt.show()