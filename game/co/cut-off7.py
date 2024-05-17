#%%
import numpy as np
from core import  calc_loss_joint, calc_cost

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
L_values = np.array(list(L.values()))
# 目标钢筋的数量
need = np.array([552, 658, 462],dtype=int)

# 禁忌搜索参数
# 最大循环次数
max_iterations = 1000000
# 禁忌表大小
tabu_tenure = 50
# 最大停滞次数
max_stagnation = 1000

# 初始化解
# patterns_length 组合的长度
# max_num 最大的组合数量
def init_solution(combination):
    return np.random.permutation(combination) 

# 评估函数
def evaluate(combinations, l, l_min):
    costs = []
    for combination in combinations:
        loss, joints = calc_loss_joint(combination, l, l_min)
        cost = calc_cost(loss, joints, l_size)
        costs.append(cost)
    return costs

# 计算小组的个数
def evaluate_groups_count(tabu_list, l, l_min):
    left_list = []
    for combination in tabu_list:
        left_list.append(get_groups_count(combination, l, l_min)[0])
    return left_list

base_combination = []
for i,key in enumerate(L):
    base_combination += [L[key]] * need[i]
patterns_length = len(base_combination)
print(f"初始化目标数量矩阵, 总数为: {patterns_length}")

# 获取小组的个数和最后一组的位置      
def get_groups_count(solution, l, l_min):
    _l = l
    count = 0
    endpos = 0
    for i,v in enumerate(solution): 
        while _l < v:
            _l += l
        _l -= v
        if _l < l_min:
            _l = l
            count += 1
            endpos = i+1
    return count, endpos

# 获得第一个小组的坐标
def get_first_group_idx(solution, l, l_min):
    _l = l
    for i,v in enumerate(solution): 
        while _l < v:
            _l += l
        _l -= v
        if _l<l_min:                            
            return i+1
    return -1

# 获得接头的个数
def get_joints(solution, l, l_min):
    joints=0
    _l = l
    for i,v in enumerate(solution): 
        while _l < v:
            _l += l
            joints+=1
        _l -= v
        if _l<l_min:                            
            _l=l
    return joints

# 邻域操作
def get_neighbor(solution):
    neighbor = np.copy(solution)  
    _, endpos = get_groups_count(neighbor,l,l_min) 
    fidx = get_first_group_idx(neighbor,l,l_min)        
    if endpos >= patterns_length-1:
        neighbor = np.concatenate((neighbor[fidx:], np.random.permutation(neighbor[:fidx])))
    else:
        neighbor = np.concatenate((neighbor[fidx:endpos], np.random.permutation(neighbor[:fidx]), np.random.permutation(neighbor[endpos:])))
    return  neighbor         

# 禁忌搜索算法
def tabu_search(max_iterations, tabu_tenure):
    # 采用随机初始解
    tabu_list = [init_solution(base_combination) for i in range(tabu_tenure)]
    tabu_waste_list = evaluate(tabu_list, l, l_min)
    tabu_groups_count_list = evaluate_groups_count(tabu_list, l, l_min)
    # 记录最佳解
    best_solution = None
    # 记录最佳解的评估
    best_waste = np.inf
    best_loss = 0
    best_joints = 0
    # 记录连续没有改进的次数
    nochange_count = 0

    for i in range(max_iterations):
        # 从禁忌表中获得一组邻域解
        neighbors=[get_neighbor(solution) for solution in tabu_list]
        # 计算邻域解的评估
        neighbors_waste_list = evaluate(neighbors, l, l_min)
        neighbors_groups_count_list = evaluate_groups_count(neighbors, l, l_min)
        
        # 选择最佳邻域解
        best_idx = np.argmin(neighbors_waste_list)
        best_neighbor, best_neighbor_waste = neighbors[best_idx], neighbors_waste_list[best_idx] 
                      
        # 禁忌搜索
        # 如果邻域解比最佳解好，更新最佳解
        if best_neighbor_waste < best_waste:
            best_solution = np.copy(best_neighbor)
            best_waste = best_neighbor_waste
            nochange_count = 0
            best_loss, best_joints = calc_loss_joint(best_solution, l, l_min)
            best_cost = calc_cost(best_loss, best_joints, l_size)
                        
        nochange_count += 1

        # 如果邻域解比当前解好，则更新禁忌组
        update_count = 0
        avg_waste = np.average(tabu_waste_list)
        avg_groups_count=np.average(tabu_groups_count_list)
        for idx, waste in enumerate(neighbors_waste_list):
            if (neighbors_groups_count_list[idx]>avg_groups_count and waste==tabu_waste_list[idx]) or (waste < avg_waste):
                update_count += 1
                tabu_list[idx]=neighbors[idx]                
                tabu_waste_list[idx]=waste
                tabu_groups_count_list[idx] = neighbors_groups_count_list[idx]       

        if i % 100 == 0:
            groups_copunt=np.average(tabu_groups_count_list)
            print(f"{i}: 禁忌组平均组个数:{groups_copunt}, 最佳成本:{best_cost}, 余料: {best_loss} 接头: {best_joints} 停滞次数: {nochange_count}/{max_stagnation}")

            # 如果连续 max_stagnation 次没有改进，则退出循环
            if nochange_count>max_stagnation:
                print("已达到目标，退出循环")
                break            

    return best_solution, best_waste

best_solution, best_waste = tabu_search(max_iterations, tabu_tenure)

# 打印最佳解决方案
loss, joints = calc_loss_joint(best_solution, l, l_min)
cost = calc_cost(loss, joints, l_size)     

out=[0,0,0]
L_values = [L[key] for key in L]
for num in base_combination:
    out[L_values.index(num)] +=1 
print(out)

print("最佳方案为：")
print("废料长度:", loss)
print("接头数量:", joints)
print("总成本:", cost)