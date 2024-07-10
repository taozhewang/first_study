import numpy as np
import copy
# # 不适合L中带有特别大的Li的情况（Li几乎等于原料）

# def decomposition2(l, L, n, length, cut, paste, accumulator, path_accumulator, path_left, pointer, stage):
#     L_length = list(L.values())
#     if pointer == len(L_length) - 1: # 如果当前pointer已经指向了最后一种L，那么不再移动pointer
#         if stage == n:               # 如果pattern总长度已经超过了n（也就是radius）倍的原长，则停止计算
#             if len(path_accumulator):   

#                 print(path_accumulator)           # 在这里统计同一个路径上产生的大pattern的种类

#                 for pattern in path_accumulator:    # path_accumulator里的第一个元素有可能是小pattern
#                     if pattern not in patterns_path:    # 对于是大pattern的情况，会有另外的path_accumulator统计到
#                         patterns_path.append(pattern)
#             return
#         Re_accumulator = copy.deepcopy(accumulator)    
#         Re_path_accumulator = copy.deepcopy(path_accumulator)
#         Re_path_left = copy.deepcopy(path_left)

#         length += L_length[pointer]
#         Re_accumulator[pointer] += 1
#         if length > l:          # 采用的是首尾相接的办法
#             length = length - l # 将length限制在0<=length<l间
#             stage += 1          # 用stage表示已经用完的l数目
#             cut += 1            # 只有在length正好到0的时候才不会让cut增加
#             paste += 1          # paste同上
#             decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
#         elif length == l:
#             length = 0
#             patterns_left.append(Re_accumulator)
#             patterns_right.append([0, stage + 1, cut, paste])
#             if 0 in Re_path_left:
#                 Re_path_accumulator.append(Re_accumulator)
#             Re_path_left.append(0)
#             stage += 1
#             pointer = 0 # 在left为0的情况下将pointer重置到0是为了在原有pattern上产生新的pattern
#                         # 在后面需要排除可以被两个独立pattern组成的大pattern的情况
#                         # 为了做到这一点需要统计可以被拆成两个独立pattern的大pattern

#                         # 所以统计在每一个pattern产生过程中是否有小pattern产生
#                         # 如果不重置pointer，大的pattern的产生路径中可能不会产生小pattern
#                         # 因为排序的不同会影响大pattern能否被拆出小pattern
#             decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
#         else:
#             # 只有在pattern总长正好抵达l的整数倍时，才能节省一次断料cut
#             cut += 1
#             left = l - length
#             if left <= losses1:
#                 # 在当前合成长度与原料截断处的距离left小于losses1时，
#                 # 记录下当前的pattern组成、原料个数stage、断料次数cut、接头用量paste
#                 patterns_left.append(Re_accumulator)
#                 patterns_right.append([left, stage + 1, cut, paste])
#                 Re_path_left.append(left)
#                 Re_path_accumulator.append(Re_accumulator)
                

#                 decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
#             else:
#                 decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
#     else:
#         if stage == n:
#             if len(path_accumulator):

#                 print(path_accumulator)

#                 for pattern in path_accumulator:
#                     if pattern not in patterns_path:
#                         patterns_path.append(pattern)
#             return
#         Re_accumulator = copy.deepcopy(accumulator)
#         Re_path_accumulator = copy.deepcopy(path_accumulator)
#         Re_path_left = copy.deepcopy(path_left)
#         decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer + 1, stage)

#         length += L_length[pointer]
#         Re_accumulator[pointer] += 1
#         if length > l:
#             length = length - l
#             stage += 1
#             cut += 1
#             paste += 1
#             decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
#         elif length == l:
#             length = 0
#             patterns_left.append(Re_accumulator)
#             patterns_right.append([0, stage + 1, cut, paste])
#             if 0 in Re_path_left:
#                 Re_path_accumulator.append(Re_accumulator)
#             Re_path_left.append(0)
#             stage += 1
#             pointer = 0
#             decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
#         else:
#             cut += 1
#             left = l - length
#             if left <= losses1:
#                 patterns_left.append(Re_accumulator)
#                 patterns_right.append([left, stage + 1, cut, paste])
#                 Re_path_left.append(left)
#                 Re_path_accumulator.append(Re_accumulator)
                

#                 decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
#             else:
#                 decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)


        
# def patterns_simplify(patterns_left, patterns_right):
#     # 用来缩减重复pattern的数量，同时去除同样的pattern但cut，paste更多的pattern
#     # 同时将产生的排序不同但组成相同的pattern简化到一种作为代表
#     patterns_left_plus, patterns_right_plus = [], []
#     k = len(patterns_left)
#     for i in range(k):
#         patl = patterns_left[i]
#         if patl not in patterns_left_plus:
#             patterns_left_plus.append(patl)
#             patterns_right_plus.append(patterns_right[i])
#         else:
#             origin_index = patterns_left_plus.index(patl)
#             cut = patterns_right_plus[origin_index][2]

#             if patterns_right[i][2] < cut:  
#                 # 通过cut多少筛选pattern
#                 patterns_left_plus.pop(origin_index)
#                 patterns_right_plus.pop(origin_index)
#                 patterns_left_plus.append(patl)
#                 patterns_right_plus.append(patterns_right[i])
#     patterns = list(zip(patterns_left_plus, patterns_right_plus))
#     patterns_main, patterns_property = {}, {}
#     for i in range(len(patterns_left_plus)):
#         patterns_main[i] = patterns_left_plus[i]
#         patterns_property[i] = patterns_right_plus[i]
#     return patterns, patterns_main, patterns_property

# def patterns_repeated(patterns, patterns_path):
#     p = []
#     for i in patterns:
#         p.append(i[0])
#     for paths in patterns_path:
#         loc = p.index(paths)
#         patterns.pop(loc)
#         p.pop(loc)
#     return patterns

# def patterns_decomposition(pattern, l, L, joint, length, count, pointer, stage):
# # 通过遍历寻找pattern的不同组成方法
# # 加入了joint来约束余料长度，顺便剪枝

#         p_length = len(pattern)
#         L_length = list(L.values())
#         Re_pattern = copy.deepcopy(pattern)
#         Re_count = copy.deepcopy(count)

#         if Re_pattern[pointer] == 0:
#             pointer = (pointer + 1) % p_length
#         else:
#             length += L_length[pointer]
            

#             if length > l:
#                 length -= l
#                 a = Re_count[stage]

#                 left = L_length[pointer] - length
#                 if left < joint or length < joint:
#                     return
                
#                 a.append(left)
#                 Re_count[stage] = a
#                 stage += 1
#                 Re_count[stage] = [length]
#                 Re_pattern[pointer] = Re_pattern[pointer] - 1

#                 if not any(Re_pattern):
#                     if l == length:
#                         Re_count.pop(stage)
#                         accumulator.append(Re_count)
#                         return
#                     else:
#                         left = l - length
#                         a = Re_count[stage]
#                         a.append(left)
#                         Re_count[stage] = a
#                         accumulator.append(Re_count)
#                         return

                                     
#                 else:
#                     for step in range(p_length):
#                         Re_pointer = step
#                         if Re_pattern[Re_pointer] != 0:
#                             patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)

#             elif length == l:
#                 length = 0
#                 a = Re_count[stage]
#                 a.append(L_length[pointer])
#                 Re_count[stage] = a
#                 stage += 1
#                 Re_count[stage] = []
#                 Re_pattern[pointer] = Re_pattern[pointer] - 1

#                 if not any(Re_pattern):
#                     Re_count.pop(stage)
#                     accumulator.append(Re_count)
#                     return
                    
#                 else:
#                     for step in range(p_length):
#                         Re_pointer = step
#                         if Re_pattern[Re_pointer] != 0:
#                             patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)

#             else:
#                 a = Re_count[stage]
#                 a.append(L_length[pointer])
#                 Re_count[stage] = a
#                 Re_pattern[pointer] = Re_pattern[pointer] - 1

#                 if not any(Re_pattern):
#                     left = l - length
#                     a = Re_count[stage]
#                     a.append(left)
#                     Re_count[stage] = a
#                     accumulator.append(Re_count)
#                     return
            
#                 else:
#                     for step in range(p_length):
#                         Re_pointer = step
#                         if Re_pattern[Re_pointer] != 0:
#                             patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)
# def patterns_decomposition_summon(pattern, l, L, joint):
#     p_length = len(pattern)
#     for i in range(p_length):
#         if pattern[i] != 0:
#             patterns_decomposition(pattern, l, L, joint, 0, {0: []}, i, 0)
#     return

# fill = input('主动填入数据请按0, 否则按照默认随便按')
# if fill == '0':
#     l = int(input('原料长度 raw material length: '))
#     n = int(input('目标材料种类数 the number of objects: '))
#     L = {}
#     for i in range(n):
#         L[i] = int(input(f'第{i + 1}种目标材料 object L{i + 1}: '))
#     radius = int(input('形成组合最多允许多少根原材料 radius of the number of raw materials: '))
#     losses1 = int(input('形成组合最多允许多长的余料 max left of patterns: '))
# else:
#     l = 12000
#     L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
#     radius = 10
#     losses1 = 0

# patterns_left = []
# patterns_right = []
# patterns_path = []
# cut = 0
# paste = 0
# accumulator = [0 for _ in L.keys()]
# pointer = 0
# stage = 0
# length = 0
# path_accumulator = []
# path_left= []
# decomposition2(l, L, radius, length, cut, paste, accumulator, path_accumulator, path_left, pointer, stage)
# print(patterns_path)

# patterns, patterns_main, patterns_property = patterns_simplify(patterns_left, patterns_right)
# patterns = patterns_repeated(patterns, patterns_path)
# # 会出现pattern1+pattern2=pattern3的情况，虽然说在后续合成过程中不会影响结果，但是会使运算上升一个维度
# # 但目前没有很好的处理这样的pattern3的方法
# # 已处理：方法如decompostion2()显示
# patterns_property = []
# patterns_ = []
# for i, pattern in enumerate(patterns):
#     print(i, pattern)
#     patterns_property.append(np.array(pattern[1]))
#     patterns_.append(np.array(pattern[0]))

# patterns = patterns_
# # patterns = list(patterns_main.values())
# # print(patterns)


# print(patterns)
# print(patterns_property)
# # # patterns = []

# patterns = []
# patterns_property = []
# def patterns_generate(l, L, joint, radius, losses, length, accumulator, stage, waste, paste, pointer, group):
#         print(group)
#         for j in range(len(group)):
#             if np.array_equal(accumulator, group[j][0]) and waste == group[j][1] and paste == group[j][2]:
#                 return
#         group.append([accumulator, waste, paste])
        

#         i = pointer
#         print(stage)

#         clength = length
#         cwaste = waste
#         cpaste = paste
#         cstage = stage

#         clength += L[i]

#         if clength >= l + joint:

#             print('case1')

#             cstage += 1
#             if cstage == radius: 

#                 print('case1 finished')

#                 return
#             else:

#                 caccumulator = copy.deepcopy(accumulator)

#                 clength -= l
#                 caccumulator[i] += 1

#                 for pointer in range(len(L)):
#                     patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, group)

#         elif clength > l:

#             print('case2')

#             cstage += 1
#             if cstage == radius: 

#                 print('case2 finished')

#                 return
#             else:

#                 caccumulator = copy.deepcopy(accumulator)

#                 cwaste += joint - (clength - l)
#                 cpaste += 1
#                 clength = joint
#                 caccumulator[i] += 1
#                 for pointer in range(len(L)):
#                     patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, group)

#         elif clength == l:

#             print('case3')

#             caccumulator = copy.deepcopy(accumulator)

#             caccumulator[i] += 1

#             for pattern in patterns:
#                 if np.array_equal(pattern, caccumulator):
#                     return
#             patterns.append(caccumulator)
#             patterns_property.append([cwaste, cpaste])

#             print('case3 finished')

#             return

#         elif clength > l - joint:

#             print('case4')

#             caccumulator = copy.deepcopy(accumulator)

#             if clength >= l - losses:
#                 cwaste += l - clength
#                 caccumulator[i] += 1

#             for pattern in patterns:
#                 if np.array_equal(pattern, caccumulator):
#                     return  
#             cwaste += l - clength
#             clength = 0
#             patterns.append(caccumulator)
#             patterns_property.append([cwaste, cpaste])
                
#             clength = 0
#             cstage += 1
#             if cstage == radius: 

#                 print('case4 finished')

#                 return
#             else:
#                 for pointer in range(len(L)):
#                     patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, group)

#         elif clength >= l - np.max(L):

#             print('case5')

#             caccumulator = copy.deepcopy(accumulator)

#             caccumulator[i] += 1
#             if clength >= l - losses:
#                 for pattern in patterns:
#                     if np.array_equal(pattern, caccumulator):
#                         return
#                 patterns.append(caccumulator)
#                 patterns_property.append([l - clength, cpaste])

#             for pointer in range(len(L)):
#                 patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, group)

#         else:

#             print('case6')

#             caccumulator = copy.deepcopy(accumulator)

#             caccumulator[i] += 1
#             if clength >= l - losses:
#                 if caccumulator in patterns:
#                     return
#                 else:
#                     patterns.append(caccumulator)
#                     patterns_property.append([l - clength, cpaste])

#             for pointer in range(i, len(L)):
#                 patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, group)

# patterns = []
# patterns_property = []
# group = []
# def patterns_generate(l, L, joint, radius, losses, length, accumulator, stage, waste, paste, pointer):

#         i = pointer
#         print(stage)

#         caccumulator = copy.deepcopy(accumulator)
#         clength = length
#         cwaste = waste
#         cpaste = paste
#         cstage = stage

#         clength += L[i]
#         caccumulator[i] += 1

#         if clength >= l + joint:
#             cstage += 1
#             cpaste += 1
#             if cstage == radius: 
#                 return
#             else:
#                 clength -= l
                
#         elif clength > l:
#             cstage += 1
#             cpaste += 1
#             cwaste += joint - (clength - l)
#             clength = joint
#             if cstage == radius: 
#                 return
                
#         elif clength == l:
#             if cwaste <= losses:
#                 for pattern in patterns:
#                     if np.array_equal(pattern, caccumulator):
#                         return
#                 patterns.append(caccumulator)
#                 patterns_property.append([cwaste, cpaste])
#                 return

#         elif clength > l - joint:
#             cwaste += l - clength
#             if cwaste <= losses:

#                 for pattern in patterns:
#                     if np.array_equal(pattern, caccumulator):
#                         return
#                 patterns.append(caccumulator)
#                 patterns_property.append([cwaste, cpaste])
                
#             clength = 0
#             cstage += 1
            
#             if cstage == radius: 
#                 return

#         else:
#             if clength >= l - losses + cwaste:

#                 for pattern in patterns:
#                     if np.array_equal(pattern, caccumulator):
#                         return
#                 patterns.append(caccumulator)
#                 patterns_property.append([l - clength + cwaste, cpaste])

#         for j in range(len(group)):
#             if np.array_equal(caccumulator, group[j][0]) and waste == group[j][1] and paste == group[j][2]:
#                 return
#         group.append([caccumulator, cwaste, cpaste])
#         print(group)

#         for pointer in range(len(L)):
#             patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer)

patterns = []
patterns_property = []
group = []
pat = []
def patterns_generate(l, L, joint, radius, losses, length, accumulator, stage, waste, paste, pointer, waste_cost, paste_cost):

        # p = '_'.join([str(x) for x in caccumulator])
        # if p in pat:
        #     loc = pat.index(p)
        #     if waste * waste_cost + paste * paste_cost < patterns_property[loc]:
        #         patterns_property[loc] = waste * waste_cost + paste * paste_cost
        #         patterns[loc] = caccumulator
        #     return
        # pat.append(p)


        if stage == radius:
            return
        i = pointer
        # print(stage)

        caccumulator = copy.deepcopy(accumulator)
        clength = length
        cwaste = waste
        cpaste = paste
        cstage = stage

        clength += L[i]
        caccumulator[i] += 1

        if clength >= l + joint:
            cstage += 1
            cpaste += 1
            clength -= l
                
        elif clength > l:
            cstage += 1
            cpaste += 1
            cwaste += joint - (clength - l)
            clength = joint
                
        elif clength == l:
            if cwaste <= losses:

                p = '_'.join([str(x) for x in caccumulator])
                cost = waste * waste_cost + paste * paste_cost
                if p in pat:
                    loc = pat.index(p)
                    cost = waste * waste_cost + paste * paste_cost
                    if cost < patterns_property[loc][0]:
                        patterns_property[loc] = [cost, cwaste, cpaste]
                        patterns[loc] = caccumulator
                    return
                pat.append(p)

                patterns.append(caccumulator)
                patterns_property.append([cost, cwaste, cpaste])
                return

        elif clength > l - joint:
            cwaste += l - clength
            if cwaste <= losses:

                p = '_'.join([str(x) for x in caccumulator])
                cost = waste * waste_cost + paste * paste_cost
                if p in pat:
                    loc = pat.index(p)
                    cost = waste * waste_cost + paste * paste_cost
                    if cost < patterns_property[loc][0]:
                        patterns_property[loc] = [cost, cwaste, cpaste]
                        patterns[loc] = caccumulator
                    return
                pat.append(p)

                patterns.append(caccumulator)
                patterns_property.append([cost, cwaste, cpaste])
                
            clength = 0
            cstage += 1

        else:
            if clength >= l - losses + cwaste:

                p = '_'.join([str(x) for x in caccumulator])
                cost = waste * waste_cost + paste * paste_cost
                if p in pat:
                    loc = pat.index(p)
                    if cost < patterns_property[loc][0]:
                        patterns_property[loc] = [cost, cwaste, cpaste]
                        patterns[loc] = caccumulator
                    return
                pat.append(p)

                patterns.append(caccumulator)
                patterns_property.append([cost, cwaste, cpaste])

        present = '_'.join([str(x) for x in caccumulator] + [str(cwaste), str(cpaste)])
        if present in group:
            return
        group.append(present)

        # for j in range(len(group)):
        #     if np.array_equal(caccumulator, group[j][0]) and waste == group[j][1] and paste == group[j][2]:
        #         return
        # group.append([caccumulator, cwaste, cpaste])
        # # print(group)

        for pointer in range(len(L)):
            patterns_generate(l, L, joint, radius, losses, clength, caccumulator, cstage, cwaste, cpaste, pointer, waste_cost, paste_cost)

taboo_size = 100

def calc_left(left, l, joint):
    length = 0
    waste = 0
    paste = 0
    for ob, num in enumerate(left):
        if num == 0: continue
        for i in range(int(num)):
            length += L[ob]

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
    waste += l - length
    return waste, paste

def solution_initialize(need, patterns, l, L, joint, taboo_size):
    solutions = np.zeros((taboo_size, len(patterns)))
    return solutions

def solution_update(solutions, need, patterns, l, L, joint, variation_count, patterns_p):
    for i in range(np.size(solutions, 0)):
        ids = np.random.choice(np.size(solutions, 1), variation_count, replace = False, p = patterns_p)
        for idx in ids:
            k = np.random.random(1)
            if k >= 0.5:
                solutions[i][idx] += 1
            else:
                solutions[i][idx] -= 1
        solutions[i][solutions[i] < 0] = 0

        accumulate = np.zeros(len(need))
        for m, j in enumerate(solutions[i]):
            accumulate += patterns[m] * j
            
        while any(accumulate > need):
            c = np.where(solutions[i] > 0)[0]
            if len(c) == 0: break
            k = np.random.choice(c)
            if solutions[i][k] > 0: solutions[i][k] -= 1
            
    return solutions

def solution_quality(solutions, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost):
    waste = 0
    paste = 0
    accumulate = np.zeros(len(need))

    C = np.zeros(np.size(solutions, 0))
    for s in range(np.size(solutions, 0)):
        for p, num in enumerate(solutions[s]):
            waste += num * patterns_property[p][1]
            paste += num * patterns_property[p][2]
            accumulate += patterns[p] * num
        
        waste2, paste2 = calc_left(need - accumulate, l, joint)
        waste += waste2
        paste += paste2

        cost = waste * waste_cost + paste * paste_cost
        C[s] = cost
    return C

def best_solution_generate(need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost, taboo_size, variation_count=3, patterns_p=None):
    depth = 0
    solutions = solution_initialize(need, patterns, l, L, joint, taboo_size)

    taboo_list = copy.deepcopy(solutions)
    waste2, paste2= calc_left(need, l, joint)
    taboo_average_cost = waste2 * waste_cost + paste2 * paste_cost
    taboo_cost = np.ones(np.size(taboo_list, 0)) * taboo_average_cost
    best_solution = np.zeros((1, len(patterns)), dtype = int)
    min_cost = solution_quality(best_solution, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost)

    while True:
        solutions = solution_update(solutions, need, patterns, l, L, joint, variation_count, patterns_p)
        # print(solutions)
        C = solution_quality(solutions, need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost)
        for i in range(len(C)):
            cost = C[i]
            if cost < taboo_average_cost:
                replace = True
                for taboo_solution in taboo_list:
                    if np.array_equal(taboo_solution, solutions[i]):
                       replace = False
                       break 
                if replace:
                    taboo_list[i] = solutions[i]
                    taboo_cost[i] = cost
                    # print('/', end = '')
        
        print(taboo_cost)
        new_taboo_average_cost = np.mean(taboo_cost)
        if new_taboo_average_cost >= taboo_average_cost: 
            depth += 1        
            # print('/', end = '')
        else:
            taboo_average_cost = new_taboo_average_cost
            print(f'depth: {depth}')
            print(f'当前最佳解: {taboo_list[np.argmin(taboo_cost)]}')
            print(f'当前平均成本: {taboo_average_cost}')
            print(f'全局最小成本: {min_cost}')
            depth = 0
        
        if depth > 1000: 
            return best_solution, min_cost

        if np.min(taboo_cost) < min_cost:
            min_cost = np.min(taboo_cost)
            best_solution = taboo_list[np.argmin(taboo_cost)]

        # half_taboo_list_index = np.argsort(taboo_cost)[:taboo_size // 2]
        # half_solutions_index = np.argsort(C)[taboo_size // 2:]

        half_taboo_list_index = np.argsort(taboo_cost)[:1]
        half_solutions_index = np.argsort(C)[1:]

        for i, j in enumerate(half_taboo_list_index):
            solutions[half_solutions_index[i]] = taboo_list[j]

        # print(solutions)

        # _solutions = np.zeros_like(solutions)
        # permutation_index = np.random.permutation(np.size(solutions, 0))
        # for i, j in enumerate(permutation_index):
        #     _solutions[i] = solutions[j]
        # solutions = _solutions


        # print('////////////////////')
        # print(taboo_average_cost)

# L = list(L.values())

# need = np.zeros(len(L))
# for i in range(len(L)):
#     need[i] = int(input(f'第{i + 1}种目标材料需要数量 object L{i + 1} need: '))
# joint = int(input('接头最少允许多长: '))
# waste_cost = int(input('余料成本 waste cost: '))
# paste_cost = int(input('粘合成本 paste cost: '))
fill = input('主动填入数据请按0, 否则按照默认随便按')
if fill == '0':
    l = int(input('原料长度 raw material length: '))
    n = int(input('目标材料种类数 the number of objects: '))
    L = np.zeros(n)
    for i in range(n):
        L[i] = int(input(f'第{i + 1}种目标材料 object L{i + 1}: '))
    radius = int(input('形成组合最多允许多少根原材料 radius of the number of raw materials: '))
    losses = int(input('形成组合最多允许多长的余料 max left of patterns: '))
else:
    l = 12000
    L = [4100, 4350, 4700]
    radius = 9
    losses = 500

need = np.array([552, 658, 462], dtype = int)
joint = 200
l_size = 32
waste_cost = 0.00617 * 2000 * (l_size ** 2) / 1000
paste_cost = 10

# for i in range(len(L)):

for i in range(len(L)):
    patterns_generate(l, L, joint, radius, losses, 0, np.zeros(len(L)), 0, 0, 0, i, waste_cost, paste_cost)
print(patterns)
print(patterns_property)

patterns_p = np.array([x[0] for x in patterns_property]) / np.sum(np.array([x[0] for x in patterns_property]))
solution, cost = best_solution_generate(need, patterns, patterns_property, l, L, joint, waste_cost, paste_cost, taboo_size, 3, patterns_p)
print(f'最佳方案 solution: {solution}')
print(f'最佳方案总成本 cost: {cost}')

ac = np.zeros(len(need))
for i, j in enumerate(solution):
    print(j, patterns[i])
    ac += patterns[i] * j
# print(ac)
print(need - ac)