#%%
import numpy as np

'''
用禁忌搜索算法求解钢筋切割问题,不依赖前期的求组合

[552, 658, 462]
最佳方案为：
废料长度: 11100
接头数量: 454
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
# 目标钢筋的数量
need = np.array([552, 658, 462],dtype=int)

# 禁忌搜索参数
# 最大循环次数
max_iterations = 1000000
# 禁忌表大小
tabu_size = 100
# 最大停滞次数
max_stagnation = 1000


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

    cost_param = 0.00617*2000*(l_size**2)/1000
    cost = loss*cost_param + joint*10
    return loss, joint, cost, group_count, group_firstpos, group_endpos

# 获得邻域解
# 选择第一组，打乱顺序加入到最后一组后面，期望找到小的接头数组合；同时打乱最后一组剩下的长度的顺序，期望可以发现新的组合
def get_neighbor(combinations, group_firstpos, group_endpos):
    combinations_size = len(combinations)
    combinations_length = len(combinations[0])
    for i in range(combinations_size):
        combination, firstpos, endpos = combinations[i], group_firstpos[i], group_endpos[i]
        if endpos < combinations_length-1:   
            combinations[i] = np.concatenate((combination[firstpos:endpos], np.random.permutation(combination[:firstpos]), np.random.permutation(combination[endpos:])))
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

print("最佳方案为：")
print(best_solution)
print("废料长度:", best_loss)
print("接头数量:", best_joints)
print("总成本:", best_cost)
