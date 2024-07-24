import numpy as np
import copy
import time

patterns = [] # 用于记录所有形如[12, 4, 5]的组合 
patterns_property = [] # 用于记录所有组合的成本、尾料、粘合次数
patterns_order = [] # 用于记录所有组合的物料顺序
group = [] # 用于记录所有种类的组合，在成本一样的情况下不考虑其排序
pat = [] # 用于记录所有种类的组合的字符串形式
#
# 下面很可能是产生组合的什么东西
# def patterns_generate(l, L, joint, radius, losses, length, accumulator, stage, waste, paste, pointer, waste_cost, paste_cost, order):

#         if stage == radius: # 如果到达最大原料用量，那么前面需要停止
#             return
#         i = pointer

#         caccumulator = copy.deepcopy(accumulator) # 用于记录当前组合
#         corder = copy.deepcopy(order) # 用于记录当前组合的物料顺序
#         clength = length # 用于记录当前组合的长度，被模l处理过了
#         cwaste = waste # 用于记录当前排序下的尾料
#         cpaste = paste # 用于记录当前排序下的粘合次数
#         cstage = stage # 用于记录当前排序下的原料用量根数

#         # 对当前组合进行加一根目标材料的处理
#         clength += L[i] 
#         caccumulator[i] += 1
#         corder.append(L[i])

#         if clength >= l + joint: # 如果当前长度大于原料长和接头长的和，那么直接切割
#             cstage += 1
#             cpaste += 1
#             clength -= l
                
#         elif clength > l: # 如果当前长度大于原料长，但多余的部分小于街头长，那么切割，同时补上一部分尾料
#             cstage += 1
#             cpaste += 1
#             cwaste += joint - (clength - l)
#             clength = joint
                
#         elif clength == l: # 如果当前长度等于原料长，直接停止
#             if cwaste <= losses:

#                 p = '_'.join([str(x) for x in caccumulator]) # 组合的字符串形式
#                 cost = cwaste * waste_cost + cpaste * paste_cost # 组合的成本
#                 if p in pat:                                        # 如果该组合已经被登记过
#                     loc = pat.index(p)                                # 找到该组合的位置    
#                     if cost < patterns_property[loc][0]:             # 如果该组合的成本更低，则更新此组合
#                         patterns_property[loc] = [cost, cwaste, cpaste]
#                         patterns[loc] = caccumulator
#                         patterns_order[loc] = corder
#                     return
#                 pat.append(p) # 登记组合

#                 patterns.append(caccumulator)
#                 patterns_property.append([cost, cwaste, cpaste])
#                 patterns_order.append(corder)
#             return

#         elif clength > l - joint: # 如果当前长度还没到原料长，但剩余长度小于接头长，则补上尾料并让长度归零
#             cwaste += l - clength
#             if cwaste <= losses:

#                 p = '_'.join([str(x) for x in caccumulator])
#                 cost = cwaste * waste_cost + cpaste * paste_cost
#                 if p in pat:
#                     loc = pat.index(p)
#                     cost = cwaste * waste_cost + cpaste * paste_cost
#                     if cost < patterns_property[loc][0]:
#                         patterns_property[loc] = [cost, cwaste, cpaste]
#                         patterns[loc] = caccumulator
#                         patterns_order[loc] = corder
#                     return
#                 pat.append(p)

#                 patterns.append(caccumulator)
#                 patterns_property.append([cost, cwaste, cpaste])
#                 patterns_order.append(corder)
                
#             clength = 0
#             cstage += 1

#         else:
#             if clength >= l - losses + cwaste:
                
#                 ccwaste = l - clength + cwaste
#                 p = '_'.join([str(x) for x in caccumulator])
#                 cost = ccwaste * waste_cost + cpaste * paste_cost
#                 if p in pat:
#                     loc = pat.index(p)
#                     if cost < patterns_property[loc][0]:
#                         patterns_property[loc] = [cost, ccwaste, cpaste]
#                         patterns[loc] = caccumulator
#                         patterns_order[loc] = corder
#                     return
#                 pat.append(p)

#                 patterns.append(caccumulator)
#                 patterns_property.append([cost, ccwaste, cpaste])
#                 patterns_order.append(corder)

#         present = '_'.join([str(x) for x in caccumulator] + [str(cwaste), str(cpaste)]) # 用于登记当前状态：组合+尾料+粘合次数
#         if present in group:                                                            # 如果该状态已经被登记过，则跳过，用于剪枝减少计算量
#             return
#         group.append(present)

#         for pointer in range(len(L)):                                                   # 遍历所有可能的组合
#             patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, waste_cost, paste_cost, corder)

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
            # print(corder)
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

population_size = 1000

def population_generate():
    population = np.zeros((population_size, len(patterns)), dtype=int)
    return population

def population_variation(population, population_size, variation_rate, patterns_p, need):
    for i in range(population_size):
        ids = np.random.choice(len(patterns), variation_rate, replace = False, p = patterns_p)
        for idx in ids:
            k = np.random.random(1)
            if k < 0.5:
                population[i][idx] -= 1
            else:
                population[i][idx] += 1
        population[i][population[i] < 0] = 0

        accumulate = np.zeros(len(need))
        for j, num in enumerate(population[i]):
            accumulate += num * patterns[j]
        
        while any(accumulate > need):
            loc = np.where(population[i] > 0)
            k = np.random.choice(loc[0])

            accumulate -= patterns[k]
            population[i][k] -= 1

    return population

# def calc_left(left, l, L, joint, waste_cost, paste_cost): # 用于计算剩余需要填补的目标材料产生的成本
#     length = 0
#     waste = 0
#     paste = 0
#     for idx, num in enumerate(left):                    # 按照顺序排序计算成本，并非最小成本
#         if num == 0: continue
#         for i in range(int(num)):
#             length += L[idx]

#             if length >= l + joint: 
#                 paste += 1
#                 length -= l
#             elif length > l:
#                 paste += 1
#                 waste += joint - (length - l)
#                 length = joint
#             elif length == l:
#                 length = 0
#             elif length > l - joint:
#                 waste += l - length
#                 length = 0
#             else:
#                 continue
#     waste += (l - length) * bool(length)
#     cost = waste * waste_cost + paste * paste_cost
#     return cost

def calc_left(left, l, L, joint, waste_cost, paste_cost): # 用于计算剩余需要填补的目标材料产生的成本
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

def population_quality(population, population_size, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost): # 计算当前解的成本
    C = np.zeros(population_size)
    W = np.zeros(np.size(population, 0))
    P = np.zeros(np.size(population, 0))
    for s in range(population_size):
        accumulate = np.zeros(len(need))
        for p, num in enumerate(population[s]):
            C[s] += num * patterns_property[p][0]
            W[s] += num * patterns_property[p][1]
            P[s] += num * patterns_property[p][2]
            accumulate += patterns[p] * num
        c, w, p = calc_left(need - accumulate, l, L, joint, waste_cost, paste_cost)
        # print(W, w)
        C[s] += c
        W[s] += w
        P[s] += p
    return C, W, P

def population_inheritance(population, population_size, C, patterns_p, need):
    parents_index = np.argsort(C)[ : population_size // 2]
    parents_cost = C[parents_index]
    parents = population[parents_index]
    population[: population_size // 2] = parents

    for i in range(population_size // 2):
        pair = np.random.choice(np.arange(population_size // 2), 2, replace = False, p = parents_cost / np.sum(parents_cost))
        # print(pair)
        parent1, parent2 = parents[pair[0]], parents[pair[1]]
        gene1 = np.random.choice(len(patterns), len(patterns) // 2, replace = False, p = patterns_p)
        # gene2 = np.arange(len(patterns))
        # gene2 = gene2[gene2 not in gene1]

        
        gene2 = np.arange(len(patterns) - len(gene1))
        k = 0
        for i in range(len(patterns)):
            if i not in gene1:
                gene2[k] = i
                k += 1
    
        descendent = np.zeros(len(patterns), dtype = int)
        descendent[gene1] = parent1[gene1]
        descendent[gene2] = parent2[gene2]
        # descendent += parent1[gene1] + parent2[gene2]

        accumulate = np.zeros(len(need))
        for j, num in enumerate(descendent):
            accumulate += num * patterns[j]
        
        while any(accumulate > need):
            loc = np.where(descendent > 0)
            k = np.random.choice(loc[0])
            accumulate -= patterns[k]
            descendent[k] -= 1
    
        population[population_size // 2 + i] = descendent

    return population

def population_selection(population_size, patterns_p, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost, variation_rate):
    
    population = population_generate()

    C = np.ones(np.size(population, 0))
    c, w, Ciallo = calc_left(need, l, L, joint, waste_cost, paste_cost)
    C *= c

    best_population = population[0]
    min_cost = C[0]

    depth = 0
    while True:
        population = population_inheritance(population, population_size, C, patterns_p, need)
        population = population_variation(population, population_size, variation_rate, patterns_p, need)
        C, W, P = population_quality(population, population_size, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost)
        # population = population_inheritance(population, population_size, C, patterns_p, need)

        curr_min_cost = np.min(C)
        curr_best_index = np.argmin(C)
        curr_best_population = population[curr_best_index]
        if curr_min_cost < min_cost:
            min_cost = curr_min_cost
            best_population = copy.deepcopy(curr_best_population)
            print(f'depth: {depth}')
            print(f'min_cost: {min_cost}')
            print(f'best_population: {best_population}')
            depth = 0

        else:
            depth += 1

        if depth > 100:
            break
        
    return best_population, min_cost

def print_left(left, l, L, joint, waste_cost, paste_cost): # 用于计算剩余需要填补的目标材料产生的成本
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

variation_rate = 3
starttime = time.time()
for i in range(len(L)):
    patterns_generate(l, L, joint, radius, losses, 0, np.zeros(len(L)), 0, 0, 0, i, waste_cost, paste_cost, [[]])
# print(patterns)
# print(patterns_property)
# print(patterns_order)

patterns_p = np.array([x[0] for x in patterns_property]) / np.sum(np.array([x[0] for x in patterns_property]))
best_population, min_cost = population_selection(population_size, patterns_p, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost, variation_rate)

print(best_population, min_cost)

# ac = np.zeros(len(need))
# for i, j in enumerate(best_population):
#     if j != 0:
#         print(j, patterns[i], patterns_order[i])
#         ac += patterns[i] * j

ac = np.zeros(len(need))
t = 0
# for i, j in enumerate(best_population):
#     if j != 0:
#         t += 1
#         print(j, patterns[i], patterns_order[i])
#         ac += patterns[i] * j

#         print(f'patternsID: {t}')
#         print(f'repeated: {j}')
#         print(f'patterns cost/ waste/ paste: {patterns_property[i]}')
#         for m, n in enumerate(patterns_order[i]):
#             if n != []:
#                 print(f'{m} : {n}')
        
for i, j in enumerate(best_population):
    if j != 0:
        t += 1
        # print(j, patterns[i], patterns_order[i])
        ac += patterns[i] * j

        print(f'patternsID: {t}')
        print(f'pattern: {patterns[i]}')
        print(f'repeated: {j}')
        print(f'cost: {patterns_property[i][0]}')
        print(f'left: {patterns_property[i][1]}')
        print(f'paste: {patterns_property[i][2]}')
        
        for m, n in enumerate(patterns_order[i]):
            if n != []:
                print(f'{m} : {n}')

C = 0
W = 0
P = 0
accumulate = np.zeros(len(need))
for p, num in enumerate(best_population):
    if num != 0:
        C += num * patterns_property[p][0]
        W += num * patterns_property[p][1]
        P += num * patterns_property[p][2]
        accumulate += patterns[p] * num

print('left: ', need - ac)
lcost, lwaste, lpaste, lorder = print_left(need - ac, l, L, joint, waste_cost, paste_cost)

C += lcost
W += lwaste 
P += lpaste

print(f'总成本: {C}, 总尾料: {W}, 总粘合次数: {P}')
print(f'剩余部分成本{lcost}')
print(f'剩余部分余料{lwaste}')
print(f'剩余部分粘合次数{lpaste}')
print(f'剩余部分切割长度{lorder}')
endtime = time.time()
print(f'运行时间: {endtime - starttime} s')