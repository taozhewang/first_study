import numpy as np
import matplotlib.pyplot as plt
import time

'''
用人工蜂群算法(ABC)求解钢筋切割问题
最佳方案为:
[4350 4100 4350 ... 4700 4700 4700]
废料长度: 11100
接头数量: 454
总成本: 144801.376
用时: 242.83246397972107 秒
'''

# 原始钢筋长度
l = 12000
# 最小可接长度,小于200的部分会做为废料
l_min = 200
# 钢筋的规格
l_size = 32
# 目标钢筋长度
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# 目标钢筋的数量
need = np.array([552, 658, 462],dtype=int)

# ABC参数
colony_size = 200        # 蜂群数量
onlooker_size = 50       # 跟随蜂数量
max_iterations = 1000    # 最大迭代次数
limit = 100              # 限制次数
max_stagnation  = 200    # 限制最大停滞次数

gen_values = []  # 记录每代的最低适应度
gen_times = []   # 记录每代的时间
start_time = time.time()

# 计算解的适应度值(成本)
def calculate_fitness(solution):
    loss, joint, cost, group_count, _, _ = evaluate(solution, l, l_min, l_size)
    return cost+1./group_count

# 计算组合的损失、接头、成本、小组个数、最后一组的位置
def evaluate(solution, l, l_min, l_size):
    _l = l
    group_count = 0
    group_endpos = 0
    loss = 0
    joint = 0
    cost = 0

    for i,num in enumerate(solution):
        while _l< num:
            _l += l
            joint += 1
        
        _l -= num

        if _l < l_min:
            loss += _l
            group_count += 1
            group_endpos = i+1
            _l = l 

    loss += _l
    cost_param = 0.00617 * 2000 * (l_size ** 2) / 1000
    cost = loss * cost_param + joint * 10
    return loss, joint, cost, group_count, 0, group_endpos

# 蜜蜂产生新解
# 新状态为当前状态的中间部分加上前半段的随机组合和后半段的随机组合
def get_new_solution(solution, l, l_min):
    start =0
    end = 0
    _l=l
    for i,length in enumerate(solution):
        while _l < length:
            _l += l
        _l -= length
        if _l < l_min:
            if start ==  0:
                start = i+1                
            end = i+1
            _l=l
    if end < len(solution)-1:   
        return np.concatenate((solution[start:end], np.random.permutation(solution[:start]), np.random.permutation(solution[end:])))
    else:                                  # 最后一组的位置刚好在最后
        return np.concatenate((solution[start:], np.random.permutation(solution[:start])))

# 比较两个group，返回接头数量选择较小的group
def get_best_solution_group(group1, group2, l, l_min):
    joint1 = joint2 = 0

    _l = l
    for length in group1:
        while _l < length:
            _l += l
            joint1 += 1
        _l -= length
        if _l < l_min:
            _l = l

    _l = l
    for length in group2:
        while _l < length:
            _l += l
            joint2 += 1
        _l -= length
        if _l < l_min:
            _l = l

    return group1 if joint1 < joint2 else group2
    
# 产生新解的随机变换
def permutative_solution(solution, l, l_min):
    start =0
    _l=l
    new_solution = []
    for i,length in enumerate(solution):
        while _l < length:
            _l += l
        _l -= length
        if _l < l_min:
            new_solution.append(get_best_solution_group(solution[start:i+1], np.random.permutation(solution[start:i+1]), l, l_min))
            start = i+1                
            _l=l
    if start < len(solution):
        new_solution.append(np.random.permutation(solution[start:]))
    return np.concatenate(new_solution)

# 初始化蜂群
def initialize_colony(colony_size, l, need):
    colony = []
    for _ in range(colony_size):
        base_combination = []
        for i, key in enumerate(L):
            base_combination += [L[key]] * need[i]
        colony.append(np.random.permutation(base_combination))
    return colony

# 引领蜂阶段，移动当前状态到新状态
def employed_bee_phase(colony, l, fitness):
    mean_fitness = np.mean(fitness)
    new_solutions = []
    for solution in colony:
        new_solutions.append(get_new_solution(solution, l, l_min))
    new_fitness = [calculate_fitness(solution) for solution in new_solutions]
    for i in range(len(colony)):
        if new_fitness[i] < mean_fitness:   
            colony[i] = new_solutions[i]
            fitness[i] = new_fitness[i]

# 跟随蜂阶段，在当前状态附近变换group位置采蜜
def onlooker_bee_phase(colony, trials, fitness):
    mean_fitness = np.mean(fitness)
    exp_values = np.exp(-np.array(fitness)/mean_fitness)
    sum_exp = np.sum(exp_values)
    probabilities = exp_values/sum_exp
    
    for _ in range(onlooker_size):
        # 轮盘赌选择蜜源
        c = np.cumsum(probabilities)
        r = np.random.uniform(0, c[-1])
        i = np.searchsorted(c, r)
        
        # 产生新蜜源
        new_solution = permutative_solution(colony[i], l, l_min)
        new_fitness = calculate_fitness(new_solution)
        
        if new_fitness < fitness[i]:
            colony[i] = new_solution
            fitness[i] = new_fitness
            trials[i] = 0
        else:
            trials[i] += 1       

# 侦查蜂阶段
def scout_bee_phase(colony, trials, limit, fitness):
    for i in range(len(colony)):
        if trials[i] >= limit:
            colony[i] = np.random.permutation(colony[i])
            trials[i] = 0

# ABC算法主程序
def abc_algorithm(colony_size, max_iterations, l, need):
    best_solution = None
    best_fitness = np.inf
    # 记录连续没有改进的次数
    nochange_count = 0    
    # 初始化蜂群
    colony = initialize_colony(colony_size, l, need)
    # 记录每个蜂群的试验次数
    trials = np.zeros(colony_size)
    # 计算初始适应度值
    fitness = [calculate_fitness(solution) for solution in colony]

    for iteration in range(max_iterations):
        # 引领蜂阶段
        employed_bee_phase(colony, l, fitness)
        # 跟随蜂阶段
        onlooker_bee_phase(colony, trials, fitness)
        # 侦查蜂阶段
        scout_bee_phase(colony, trials, limit, fitness)
        
        fitness = [calculate_fitness(solution) for solution in colony]

        current_best_fitness = min(fitness)
        current_best_solution = colony[fitness.index(current_best_fitness)]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            best_loss, best_joints, best_cost, group_count, _, _ = evaluate(current_best_solution, l, l_min, l_size)
            nochange_count = 0
        else:
            nochange_count += 1

        gen_values.append(best_cost)
        gen_times.append(time.time()-start_time)

        if iteration % 10 == 0:
            print(f"{iteration}: 全局最佳适应度 = {best_fitness}, 本轮最佳适应度 = {current_best_fitness}, 全局最佳成本 = {best_cost}, 停滞次数: {nochange_count}/{max_stagnation}")
    
         # 限制最大停滞次数
        if nochange_count >= max_stagnation:
            break

    return best_solution, best_fitness

best_solution, best_fitness = abc_algorithm(colony_size, max_iterations, l, need)

out = [0, 0, 0]
L_values = [L[key] for key in L]
for num in best_solution:
    out[L_values.index(num)] += 1
print("验算结果:", out, "实际需求:", need)

best_loss, best_joints, best_cost, _, _, _ = evaluate(best_solution, l, l_min, l_size)
print("最佳方案为:")
print(best_solution)
print("废料长度:", best_loss)
print("接头数量:", best_joints)
print("总成本:", best_cost)
print(f"用时: {time.time() - start_time} 秒")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.plot(gen_times, gen_values, marker='o', label='最低成本')
plt.xlabel('时间(s)')
plt.ylabel('目标函数值')
plt.title('蜂群算法(ABC)收敛速度图')
plt.grid(True)
plt.legend()
plt.show()