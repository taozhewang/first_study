#%%
import numpy as np
import matplotlib.pyplot as plt
import time

'''
用禁忌搜索算法求解钢筋切割问题,不依赖前期的求组合

最佳方案为：
51 * [4, 4, 3] loss: 100 : [4100, 4100, 4350, 4100, 4350, 4350, 4700, 4100, 4350, 4700, 4700]
15 * [7, 5, 2] loss: 150 : [4100, 4100, 4700, 4350, 4100, 4350, 4350, 4100, 4700, 4100, 4100, 4100, 4350, 4350]
38 * [1, 3, 4] loss: 50 : [4350, 4700, 4700, 4350, 4350, 4700, 4700, 4100]
19 * [5, 2, 4] loss: 0 : [4100, 4700, 4100, 4350, 4700, 4700, 4100, 4100, 4700, 4100, 4350]
18 * [5, 8, 1] loss: 0 : [4350, 4350, 4350, 4100, 4100, 4350, 4700, 4100, 4100, 4100, 4350, 4350, 4350, 4350]
1 * [8, 3, 3] loss: 50 : [4350, 4100, 4100, 4100, 4100, 4100, 4100, 4700, 4700, 4100, 4350, 4100, 4700, 4350]
3 * [1, 9, 1] loss: 50 : [4350, 4350, 4700, 4350, 4350, 4350, 4100, 4350, 4350, 4350, 4350]
1 * [4, 10, 0] loss: 100 : [4100, 4350, 4350, 4350, 4350, 4350, 4350, 4100, 4350, 4100, 4100, 4350, 4350, 4350]
8 * [0, 5, 3] loss: 150 : [4700, 4700, 4700, 4350, 4350, 4350, 4350, 4350]
1 * [5, 3, 3] loss: 350 : [4350, 4350, 4100, 4100, 4100, 4350, 4700, 4700, 4700, 4100, 4100]
废料长度: 11100.0
接头数量: 454
总成本: 144801.376
用时: 62.27786350250244 秒
'''

# 原始钢筋长度
l = 12000
# 最小可接长度，小于200的部分会做为废料
l_min = 200
# 钢筋的规格
l_size = 32
# 目标钢筋长度
L = {'L1' : 3110, 'L2' : 4340, 'L3' : 5310}
# 目标钢筋的数量
need = np.array([852, 658, 162],dtype=int)

# 禁忌搜索参数
# 最大循环次数
max_iterations = 1000000
# 禁忌表大小
tabu_size = 100
# 最大停滞次数
max_stagnation = 1000

gen_values = []  # 记录每代的最低适应度
gen_times = []   # 记录每代的时间

start_time = time.time()
# 计算组合的损失、接头、成本、小组个数、最后一组的位置
def evaluate(combinations, l, l_min, l_size):
    combinations_size = len(combinations)                           # 禁忌表大小
    _l = np.ones(combinations_size)*l                               # 剩余长度
    group_count = np.zeros(combinations_size,dtype=int)             # 小组个数
    group_firstpos = np.zeros(combinations_size,dtype=int)          # 第一个小组的位置
    group_endpos = np.zeros(combinations_size,dtype=int)            # 最后一个小组的位置
    loss = np.zeros(combinations_size,dtype=float)                  # 余料
    joint = np.zeros(combinations_size,dtype=int)                   # 接头
    cost = np.zeros(combinations_size,dtype=float)                  # 成本

    for i in range(len(combinations[0])): 
        while True:
            idxs=np.where(_l<combinations[:,i])[0]
            if len(idxs)==0: break
            _l[idxs] += l
            joint[idxs] += 1
        _l -= combinations[:,i]

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

        # 确定其他
        idxs=np.where(_l<l_min)[0]
        if len(idxs)>0:
            loss[idxs] += _l[idxs]
            group_count[idxs] += 1
            group_endpos[idxs] = i+1
            _l[idxs] = l

    loss += _l

    cost_param = 0.01254#0.00617*2000*(l_size**2)/1000
    cost = loss*cost_param + joint*9#10
    return loss, joint, cost, group_count, group_firstpos, group_endpos

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
        if _l < l_min:
            _l = l

    _l = l
    for length in group2:
        while _l < length:
            _l += l
            joint2 += 1
        _l -= length
        if l-_l<l_min: _l -= l_min-(l-_l)
        if _l < l_min:
            _l = l

    return group1 if joint1 < joint2 else group2

# 获得邻域解
# 选择第一组，打乱顺序加入到最后一组后面，期望找到小的接头数组合；同时打乱最后一组剩下的长度的顺序，期望可以发现新的组合
def get_neighbor(combinations, group_firstpos, group_endpos):
    combinations_size = len(combinations)
    combinations_length = len(combinations[0])
    for i in range(combinations_size):
        combination, firstpos, endpos = combinations[i], group_firstpos[i], group_endpos[i]
        if endpos < combinations_length-1:  
            first_group = get_best_solution_group(combination[:firstpos], np.random.permutation(combination[:firstpos]), l, l_min)
            combinations[i] = np.concatenate((combination[firstpos:endpos], first_group, np.random.permutation(combination[endpos:])))
        else:                                  # 最后一组的位置刚好在最后
            combinations[i] = np.concatenate((combination[firstpos:], np.random.permutation(combination[:firstpos])))
    return combinations

# 禁忌搜索算法
def tabu_search(max_iterations, tabu_size):

    base_combination = []
    for i,key in enumerate(L):
        base_combination += [L[key]] * need[i]    

    # 采用随机初始解
    tabu_list = np.array([np.random.permutation(base_combination) for _ in range(tabu_size)])
    # 计算初始解的评估
    tabu_loss, tabu_joint, tabu_cost, tabu_group_count, group_firstpos, group_endpos = evaluate(tabu_list, l, l_min, l_size)
    # 记录最佳解
    best_solution = None
    # 记录最佳解的评估
    best_cost = np.inf
    best_loss = 0
    best_joints = 0
    # 记录连续没有改进的次数
    nochange_count = 0

    for i in range(max_iterations):
        # 从禁忌表中获得一组邻域解
        neighbors = get_neighbor(tabu_list, group_firstpos, group_endpos)
        # 计算邻域解的评估
        neighbors_loss, neighbors_joint, neighbors_cost, neighbors_group_count, group_firstpos, group_endpos = evaluate(neighbors, l, l_min, l_size)
        
        # 选择最佳邻域解
        best_idx = np.argmin(neighbors_cost)
        best_neighbor_cost = neighbors_cost[best_idx] 
                      
        # 禁忌搜索
        # 如果邻域解比最佳解好，更新最佳解
        if best_neighbor_cost < best_cost:
            best_solution = np.copy(neighbors[best_idx])
            best_cost = best_neighbor_cost
            nochange_count = 0
            best_loss = neighbors_loss[best_idx]
            best_joints = neighbors_joint[best_idx]
                        
        gen_values.append(best_cost)
        gen_times.append(time.time()-start_time)

        nochange_count += 1

        # 如果邻域解比当前解好，则更新禁忌组
        update_count = 0
        avg_waste = np.average(tabu_cost)
        avg_groups_count=np.average(tabu_group_count)
        for idx, waste in enumerate(neighbors_cost):
            if (neighbors_group_count[idx]>avg_groups_count) or (waste < avg_waste):
                update_count += 1
                tabu_list[idx]=neighbors[idx]                
                tabu_cost[idx]=waste
                tabu_group_count[idx] = neighbors_group_count[idx]       

        if i % 100 == 0:
            groups_copunt=np.average(tabu_group_count)
            print(f"{i}: 禁忌组平均组个数:{groups_copunt}, 最佳成本:{best_cost}, 余料: {best_loss} 接头: {best_joints} 停滞次数: {nochange_count}/{max_stagnation}")

            # 如果连续 max_stagnation 次没有改进，则退出循环
            if nochange_count>max_stagnation:
                print("已达到目标，退出循环")
                break            

    return best_solution, best_cost, best_loss, best_joints

best_solution, best_cost, best_loss, best_joints = tabu_search(max_iterations, tabu_size)

out=[0,0,0]
L_values = [L[key] for key in L]
for num in best_solution:
    out[L_values.index(num)] +=1 
print("验算结果:", out, "实际需求:", need)

# 将排序转为group key:[]
counters=[]
groups=[]
nums=[]
loss=[]
_group=[]
_l=l
names=list(L.keys())
for p in best_solution:
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

print("废料长度:", best_loss)
print("接头数量:", best_joints)
print("总成本:", best_cost)
print(f"用时: {time.time() - start_time} 秒")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.plot(gen_times, gen_values, marker='o', label='最低成本')
plt.xlabel('时间(s)')
plt.ylabel('目标函数值')
plt.title('禁忌搜索算法收敛速度图')
plt.grid(True)
plt.legend()
plt.show()