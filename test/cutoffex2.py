'''
最佳方案总成本 cost: 144801.37599999996
27.0 [8. 3. 3.] [4100, 4100, 4100, 4100, 4100, 4100, 4100, 4100, 4350, 4350, 4350, 4700, 4700, 4700]
3.0 [5. 2. 4.] [4100, 4100, 4100, 4100, 4100, 4350, 4350, 4700, 4700, 4700, 4700]
64.0 [4. 4. 3.] [4100, 4100, 4100, 4100, 4350, 4350, 4350, 4350, 4700, 4700, 4700]
22.0 [1. 9. 1.] [4100, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4700]
38.0 [1. 3. 4.] [4100, 4350, 4350, 4350, 4700, 4700, 4700, 4700]
[5. 3. 3.]
运行时间: 52.61149525642395 s
最佳方案总成本 cost: 144801.37599999996
30.0 [8. 3. 3.] [4100, 4100, 4100, 4100, 4100, 4100, 4100, 4100, 4350, 4350, 4350, 4700, 4700, 4700]
2.0 [5. 2. 4.] [4100, 4100, 4100, 4100, 4100, 4350, 4350, 4700, 4700, 4700, 4700]
3.0 [ 4. 10.  0.] [4100, 4100, 4100, 4100, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350]
57.0 [4. 4. 3.] [4100, 4100, 4100, 4100, 4350, 4350, 4350, 4350, 4700, 4700, 4700]
20.0 [1. 9. 1.] [4100, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4700]
42.0 [1. 3. 4.] [4100, 4350, 4350, 4350, 4700, 4700, 4700, 4700]
[0. 0. 5.]
运行时间: 39.02084422111511 s
最佳方案总成本 cost: 144801.37599999996
29.0 [8. 3. 3.] [4100, 4100, 4100, 4100, 4100, 4100, 4100, 4100, 4350, 4350, 4350, 4700, 4700, 4700]
1.0 [5. 8. 1.] [4100, 4100, 4100, 4100, 4100, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4700]
7.0 [ 4. 10.  0.] [4100, 4100, 4100, 4100, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350]
53.0 [4. 4. 3.] [4100, 4100, 4100, 4100, 4350, 4350, 4350, 4350, 4700, 4700, 4700]
15.0 [1. 9. 1.] [4100, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4700]
48.0 [1. 3. 4.] [4100, 4350, 4350, 4350, 4700, 4700, 4700, 4700]
[12.  2.  8.]
运行时间: 18.6979820728302 s
最佳方案总成本 cost: 144801.37599999996
33.0 [8. 3. 3.] [4100, 4100, 4100, 4100, 4100, 4100, 4100, 4100, 4350, 4350, 4350, 4700, 4700, 4700]
54.0 [4. 4. 3.] [4100, 4100, 4100, 4100, 4350, 4350, 4350, 4350, 4700, 4700, 4700]
13.0 [1. 9. 1.] [4100, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4350, 4700]
47.0 [1. 3. 4.] [4100, 4350, 4350, 4350, 4700, 4700, 4700, 4700]
[12. 85.  0.]
运行时间: 21.853185892105103 s
'''
import numpy as np
import copy
import time
import pandas as pd

patterns = [] # 用于记录所有形如[12, 4, 5]的组合 
patterns_property = [] # 用于记录所有组合的成本、尾料、粘合次数
patterns_order = [] # 用于记录所有组合的物料顺序
group = [] # 用于记录所有种类的组合，在成本一样的情况下不考虑其排序
pat = [] # 用于记录所有种类的组合的字符串形式

# 下面很可能是产生组合的什么东西
def patterns_generate(l, L, joint, radius, losses, length, accumulator, stage, waste, paste, pointer, waste_cost, paste_cost, order):

        if stage == radius: # 如果到达最大原料用量，那么前面需要停止
            return
        i = pointer

        caccumulator = copy.deepcopy(accumulator) # 用于记录当前组合
        corder = copy.deepcopy(order) # 用于记录当前组合的物料顺序
        clength = length # 用于记录当前组合的长度，被模l处理过了
        cwaste = waste # 用于记录当前排序下的尾料
        cpaste = paste # 用于记录当前排序下的粘合次数
        cstage = stage # 用于记录当前排序下的原料用量根数

        # 对当前组合进行加一根目标材料的处理
        clength += L[i] 
        caccumulator[i] += 1
        # corder.append(L[i])

        if clength >= l + joint: # 如果当前长度大于原料长和接头长的和，那么直接切割
            cstage += 1
            cpaste += 1
            clength -= l
            corder[stage].extend([l - length])
            corder.append([clength])
                
        elif clength > l: # 如果当前长度大于原料长，但多余的部分小于街头长，那么切割，同时补上一部分尾料
            cstage += 1
            cpaste += 1
            cwaste += joint - (clength - l)
            clength = joint
            corder[stage].extend([joint - (clength - l), L[i] - joint])
            corder.append([clength])
                
        elif clength == l: # 如果当前长度等于原料长，直接停止
            corder[stage].extend([L[i]])
            if cwaste <= losses:

                p = '_'.join([str(x) for x in caccumulator]) # 组合的字符串形式
                cost = cwaste * waste_cost + cpaste * paste_cost # 组合的成本
                if p in pat:                                        # 如果该组合已经被登记过
                    loc = pat.index(p)                                # 找到该组合的位置    
                    if cost < patterns_property[loc][0]:             # 如果该组合的成本更低，则更新此组合
                        patterns_property[loc] = [cost, cwaste, cpaste]
                        patterns[loc] = caccumulator
                        patterns_order[loc] = corder
                    return
                pat.append(p) # 登记组合

                patterns.append(caccumulator)
                patterns_property.append([cost, cwaste, cpaste])
                patterns_order.append(corder)
            return

        elif clength > l - joint: # 如果当前长度还没到原料长，但剩余长度小于接头长，则补上尾料并让长度归零
            cwaste += l - clength
            corder[cstage].extend([L[i], l - clength])
            corder.append([])
            if cwaste <= losses:

                p = '_'.join([str(x) for x in caccumulator])
                cost = cwaste * waste_cost + cpaste * paste_cost
                if p in pat:
                    loc = pat.index(p)
                    cost = cwaste * waste_cost + cpaste * paste_cost
                    if cost < patterns_property[loc][0]:
                        patterns_property[loc] = [cost, cwaste, cpaste]
                        patterns[loc] = caccumulator
                        patterns_order[loc] = corder
                    return
                pat.append(p)

                patterns.append(caccumulator)
                patterns_property.append([cost, cwaste, cpaste])
                patterns_order.append(corder)
                
            clength = 0
            cstage += 1

        else:
            print(corder)
            corder[cstage].extend([L[i]])
            if clength >= l - losses + cwaste:
                
                ccwaste = l - clength + cwaste
                p = '_'.join([str(x) for x in caccumulator])
                cost = ccwaste * waste_cost + cpaste * paste_cost
                if p in pat:
                    loc = pat.index(p)
                    if cost < patterns_property[loc][0]:
                        patterns_property[loc] = [cost, ccwaste, cpaste]
                        patterns[loc] = caccumulator
                        patterns_order[loc] = corder
                    return
                pat.append(p)

                patterns.append(caccumulator)
                patterns_property.append([cost, ccwaste, cpaste])
                patterns_order.append(corder)

        present = '_'.join([str(x) for x in caccumulator] + [str(cwaste), str(cpaste)]) # 用于登记当前状态：组合+尾料+粘合次数
        if present in group:                                                            # 如果该状态已经被登记过，则跳过，用于剪枝减少计算量
            return
        group.append(present)

        for pointer in range(len(L)):                                                   # 遍历所有可能的组合
            patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, waste_cost, paste_cost, corder)

taboo_size = 100 # 禁忌表的长度

def calc_left(left, l, joint, waste_cost, paste_cost): # 用于计算剩余需要填补的目标材料产生的成本
    length = 0
    waste = 0
    paste = 0
    for idx, num in enumerate(left):                    # 按照顺序排序计算成本，并非最小成本
        if num == 0: continue
        for i in range(int(num)):
            length += L[idx]

            if length >= l + joint: 
                paste += 1
                length -= l
            elif length > l:
                paste += 1
                waste += joint - (length - l)
                length = joint
            elif length == l:
                length = 0
            elif length > l - joint:
                waste += l - length
                length = 0
            else:
                continue
    waste += (l - length) * bool(length)
    cost = waste * waste_cost + paste * paste_cost
    return cost, waste, paste

def solution_initialize(need, patterns, l, L, joint, taboo_size): # 产生一个初始解
    solutions = np.zeros((taboo_size, len(patterns)))
    return solutions

def solution_update(solutions, need, patterns, l, L, joint, variation_count, patterns_p): # 更新解，即寻找解的邻域
    for i in range(np.size(solutions, 0)):
        ids = np.random.choice(np.size(solutions, 1), variation_count, replace = False, p = patterns_p) # 根据成本大小选择需要变动的组合，选择个数为variation_count
        for idx in ids:
            k = np.random.random(1) # 随机加一减一处理
            if k >= 0.5:
                solutions[i][idx] += 1
            else:
                solutions[i][idx] -= 1
        solutions[i][solutions[i] < 0] = 0 # 避免出现个数为负数的情况

        accumulate = np.zeros(len(need))
        for m, j in enumerate(solutions[i]):
            accumulate += patterns[m] * j
            
        while any(accumulate > need): # 将当前解内所有目标个数维持到需求以下
            c = np.where(solutions[i] > 0)[0]
            if len(c) == 0: break
            k = np.random.choice(c)
            if solutions[i][k] > 0: 
                solutions[i][k] -= 1
                accumulate -= patterns[k]
    return solutions

def solution_quality(solutions, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost): # 计算当前解的成本
    C = np.zeros(np.size(solutions, 0))
    W = np.zeros(np.size(solutions, 0))
    P = np.zeros(np.size(solutions, 0))
    for s in range(np.size(solutions, 0)):
        accumulate = np.zeros(len(need))
        cost = 0
        waste = 0
        paste = 0
        for p, num in enumerate(solutions[s]):
            cost += num * patterns_property[p][0]
            waste += num * patterns_property[p][1]
            paste += num * patterns_property[p][2]
            accumulate += patterns[p] * num
        
        left_cost, left_waste, left_paste = calc_left(need - accumulate, l, joint, waste_cost, paste_cost)
        cost += left_cost
        waste += left_waste
        paste += left_paste

        C[s] = cost
        W[s] = waste
        P[s] = paste
    return C, W, P

def best_solution_generate(need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost, taboo_size, variation_count=3, patterns_p=None): # 迭代产生最佳解
    depth = 0
    solutions = solution_initialize(need, patterns, l, L, joint, taboo_size)

    taboo_list = copy.deepcopy(solutions)
    taboo_average_cost, taboo_average_waste, taboo_average_paste = calc_left(need, l, joint, waste_cost, paste_cost)
    taboo_cost = np.ones(np.size(taboo_list, 0)) * taboo_average_cost

    best_solution = taboo_list[0]
    min_cost = taboo_average_cost
    min_cost_waste = taboo_average_waste
    min_cost_paste = taboo_average_paste

    while True:
        solutions = solution_update(solutions, need, patterns, l, L, joint, variation_count, patterns_p)
        # print(solutions)
        C, W, P = solution_quality(solutions, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost)
        # print(C)
        for i in range(len(C)):
            cost = C[i]
            if cost < taboo_average_cost: # 如果当前解的成本小于禁忌表平均成本，就将此解加入禁忌表
                replace = True
                for taboo_solution in taboo_list:
                    if np.array_equal(taboo_solution, solutions[i]):
                       replace = False
                       break 
                if replace: # 禁忌表中解不重复
                    taboo_list[i] = solutions[i]
                    taboo_cost[i] = cost
                    # print('/', end = '')
        
        # print(taboo_cost)
        new_taboo_average_cost = np.mean(taboo_cost)
        if new_taboo_average_cost >= taboo_average_cost: 
            depth += 1        
            # print('/', end = '')
        else:
            taboo_average_cost = new_taboo_average_cost
            print(f'depth: {depth}')
            # print(f'当前最佳解: {taboo_list[np.argmin(taboo_cost)]}')
            print(f'当前平均成本: {taboo_average_cost}')
            print(f'全局最小成本: {min_cost}')
            print(f'全局最佳解: {best_solution}')
            depth = 0
        
        if depth > 100: # 最大循环次数
            return best_solution, min_cost, min_cost_waste, min_cost_paste

        if np.min(taboo_cost) < min_cost: # 更新最佳解
            min_cost = np.min(taboo_cost)
            idx = np.argmin(taboo_cost)
            best_solution = taboo_list[idx]
            min_cost_waste = W[idx]
            min_cost_paste = P[idx]

        # 将禁忌表中的解替换当前解，并从当前解的邻域寻找更好的解

        # half_taboo_list_index = np.argsort(taboo_cost)[:taboo_size // 2] # 替换50%
        # half_solutions_index = np.argsort(C)[taboo_size // 2:]

        # half_taboo_list_index = np.argsort(taboo_cost)[:taboo_size // 10] # 替换10%
        # half_solutions_index = np.argsort(C)[taboo_size // 10:]

        half_taboo_list_index = np.argsort(taboo_cost) # 替换100%
        half_solutions_index = np.argsort(C)      

        for i, j in enumerate(half_taboo_list_index):
            solutions[half_solutions_index[i]] = taboo_list[j]

def print_left(left, l, joint, waste_cost, paste_cost): # 用于计算剩余需要填补的目标材料产生的成本
    length = 0
    waste = 0
    paste = 0
    stage = 0
    order = [[]]
    for idx, num in enumerate(left):                    # 按照顺序排序计算成本，并非最小成本
        if num == 0: continue
        for i in range(int(num)):
            length += L[idx]

            if length >= l + joint: 
                order[stage].extend([L[idx] - (length - l)])
                order.append([length - l])
                stage += 1
                paste += 1
                length -= l

            elif length > l:
                order[stage].extend([joint - (length - l), L[idx] - joint])
                order.append([joint])
                stage += 1
                paste += 1
                waste += joint - (length - l)
                length = joint

            elif length == l:
                order[stage].extend([L[idx]])
                order.append([])
                stage += 1
                length = 0

            elif length > l - joint:
                order[stage].extend([L[idx], l - length])
                order.append([])
                stage += 1
                waste += l - length
                length = 0

            else:
                order[stage].extend([L[idx]])
                continue
    if length > 0:
        order[stage].extend([l - length])
        waste += l - length
    cost = waste * waste_cost + paste * paste_cost
    return cost, waste, paste, order
    
fill = input('主动填入数据请按0, 否则按照默认随便按')
if fill == '0':
    l = int(input('原料长度 raw material length: '))
    n = int(input('目标材料种类数 the number of objects: '))
    radius = int(input('形成组合最多允许多少根原材料 radius of the number of raw materials: '))
    losses = int(input('形成组合最多允许多长的余料 max left of patterns: '))

    L = np.zeros(n)
    need = np.zeros(n)
    for i in range(n):
        L[i] = int(input(f'第{i + 1}种目标材料 object L{i + 1}: '))
        need[i] = int(input(f'第{i + 1}种目标材料需要数量 object L{i + 1} need: '))

    joint = int(input('接头最少允许多长: '))
    waste_cost = int(input('余料成本 waste cost: '))
    paste_cost = int(input('粘合成本 paste cost: '))


    
else:
    l = 12000
    L = [4100, 4350, 4700]
    radius = 5
    losses = 100

    need = np.array([852, 658, 162], dtype = int)
    joint = 200
    l_size = 32
    waste_cost = 0.00617 * 2000 * (l_size ** 2) / 1000
    paste_cost = 10
    # waste_cost = 1
    # paste_cost = 1


# for i in range(len(L)):
starttime = time.time()
for i in range(len(L)):
    patterns_generate(l, L, joint, radius, losses, 0, np.zeros(len(L)), 0, 0, 0, i, waste_cost, paste_cost, [[]])
print(patterns)
print(patterns_property)
print(patterns_order)

patterns_p = np.array([x[0] for x in patterns_property]) / np.sum(np.array([x[0] for x in patterns_property]))
solution, cost, waste, paste = best_solution_generate(need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost, taboo_size, 3, patterns_p)
print(f'最佳方案 solution: {solution}')
print(f'最佳方案总成本 cost: {cost}')
print(f'最佳方案总余料 waste: {waste}')
print(f'最佳方案总粘合 paste: {paste}')

ac = np.zeros(len(need))
t = 0
for i, j in enumerate(solution):
    if j != 0:
        t += 1
        print(j, patterns[i], patterns_order[i])
        ac += patterns[i] * j

        print(f'patternsID: {t}')
        print(f'repeated: {j}')
        for m, n in enumerate(patterns_order[i]):
            if n != []:
                print(f'{m} : {n}')


print(need - ac)
lcost, lwaste, lpaste, lorder = print_left(need - ac, l, joint, waste_cost, paste_cost)
print(f'剩余部分成本{lcost}')
print(f'剩余部分余料{lwaste}')
print(f'剩余部分粘合次数{lpaste}')
print(f'剩余部分切割长度{lorder}')
endtime = time.time()
print(f'运行时间: {endtime - starttime} s')