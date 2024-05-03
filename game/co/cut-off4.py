#%%
import numpy as np
import random

from core import pattern_oringin, calc_loss_joint, calc_cost, calc_completion_lenghts

'''
用禁忌搜索算法求解钢筋切割问题
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

# 初始化单个组合的最大数量
max_num = 1
# 最大的组合长度
radius = 10
# 组合数最小余料
losses1 = 30

# 禁忌搜索参数
# 最大循环次数
max_iterations = 1000000
# 禁忌期限
tabu_tenure = 200

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
    dl=np.array(list(L.values()))
    loss, joint = calc_loss_joint(bar_lengths, l, dl, l_min)
    cost += calc_cost(loss, joint, l_size)    
    # 计算成本和完成距离目标的距离
    cost += np.sum(np.abs(bar_lengths))*1000
    return cost

# 求各种组合的列表
patterns = pattern_oringin(l, L, losses1, radius)
patterns_length = len(patterns)
print(f"patterns[0]:", patterns[0])
print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
print(f"patterns length: {patterns_length}")# 产生patterns，最低1个组合，因为需要处理尾料

# 邻域操作
def get_neighbor(solution, patterns_length, variation_count):
    neighbor = np.copy(solution)
    ids = np.random.choice(patterns_length, variation_count, replace=False)
    # 为了加快收敛，变异方向固定
    v = 1 if random.random()<0.5 else -1
    for idx in ids:
        # 如果只剩下2个变异位置，则变异方向随机
        if variation_count==2: v = 1 if random.random()<0.5 else -1
        neighbor[idx] += v
        if neighbor[idx] < 0: neighbor[idx]= 0
    return neighbor

# 禁忌搜索,检查邻域解是否在禁忌表中
def check_tabu(tabu_list, neighbor):
    for tabu in tabu_list:
        if np.array_equal(tabu, neighbor):
            return True
    return False

# 禁忌搜索算法
def tabu_search(max_iterations, tabu_tenure, patterns_length, max_num):
    # 采用随机初始解
    tabu_list = [init_solution(patterns_length, max_num) for i in range(tabu_tenure)]
    # 初始评估
    tabu_waste_list = [evaluate(solution, need, patterns) for solution in tabu_list]

    # 记录最佳解
    best_solution = None
    # 记录最佳解的评估
    best_waste = np.inf
    # 记录连续没有改进的次数
    nochange_count = 0

    # 动态调整异动个数
    variation_count = patterns_length//8
    for i in range(max_iterations):
        # 从禁忌表中获得一组邻域解
        neighbors = [get_neighbor(solution, patterns_length, variation_count) for solution in tabu_list]
        # 计算邻域解的评估
        neighbors_waste = [evaluate(neighbor, need, patterns) for neighbor in neighbors]
        # 选择最佳邻域解
        best_neighbor, best_neighbor_waste = min(zip(neighbors, neighbors_waste), key=lambda x: x[1])       

        # 禁忌搜索
        # 如果邻域解比最佳解好，更新最佳解
        if best_neighbor_waste < best_waste:
            best_solution = best_neighbor
            best_waste = best_neighbor_waste
            nochange_count = 0
            
        nochange_count += 1

        # 如果邻域解比禁忌组的平均成本好，且邻域解不在禁忌表中，则更新禁忌组
        avg_waste = sum(tabu_waste_list)/len(tabu_waste_list)
        update_count = 0
        for idx, waste in enumerate(neighbors_waste):
            if waste < avg_waste and not check_tabu(tabu_list, neighbors[idx]):
                # 记录最佳解
                tabu_list.append(neighbors[idx])
                tabu_waste_list.append(waste)
                update_count += 1
                # 限制禁忌表的长度,删除掉最差的解
                if len(tabu_list) > tabu_tenure:
                    idx = np.argmax(tabu_waste_list)
                    tabu_list.pop(idx)
                    tabu_waste_list.pop(idx)

        if i % 10 == 0:
            best_used = calc_completion_lenghts(best_solution, need, patterns)

            # 动态调整异动个数
            variation_count = np.sum(np.abs(best_used-need))//100
            if variation_count>patterns_length//2: variation_count=patterns_length//2
            if variation_count<2: variation_count=2

            print(f"{i}: 禁忌组平均成本:{avg_waste}, 更新个数:{update_count}, 最佳成本:{best_waste}, 最佳完成度: {best_used} 目标: {need} 异动个数: {variation_count}")

            # 如果数量匹配，且连续100次没有改进，则退出循环
            if np.array_equal(best_used, need) and nochange_count>100:
                print("已达到目标，退出循环")
                break            

    return best_solution, best_waste

best_solution, best_waste = tabu_search(max_iterations, tabu_tenure, patterns_length, max_num)

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
        
print("最后结果:", bar_lengths, "目标:", need)
print("废料长度:", loss)
print("接头数量:", joint)
print("总成本:", cost)
