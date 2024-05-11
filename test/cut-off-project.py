# #################
# import numpy as np
# import copy
# l = 12000
# L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# # L = {'L2' : 4350, 'L1' : 4100, 'L3' : 4700}
# # L = {'L2' : 4350, 'L3' : 4700, 'L1' : 4100}
# # L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700, 'L4' : 3350, 'L5' : 3700, 'L6': 2100}
# need = np.array([852, 658, 162])
# radius = 4
# losses1 = 0
# losses2 = 40

# '''patterns: [([5, 2, 4], [0, 4, 10, 3]), ([number1, number2, number3], [left, ls, cut, paste])]
# cut, paste: integers
# count: length
# accumulator: [5, 2, 4] ~ the number of L1 L2 L3

# '''
# patterns_left = []
# patterns_right = []
# patterns_turn = []
# cut = 0
# paste = 0
# count = 0
# accumulator = [0 for _ in L.keys()]
# turn_accumulator = []
# pointer = 0
# def decomposition1(l, L, n, count, cut, paste, accumulator, turn_accumulator, pointer):
#     L_values = list(L.values())
#     stage = count // l
#     count += L_values[pointer]
#     Re_stage, Re_end_of_the_stage = count // l, count % l
#     Re_end_of_the_stage = l * (Re_stage + bool(Re_end_of_the_stage)) - count
#     Re_accumulator = copy.deepcopy(accumulator)
#     Re_turn_accumulator = copy.deepcopy(turn_accumulator)
#     if pointer == len(L_values) - 1:
#         if Re_stage == n and Re_end_of_the_stage == 0:
#             Re_accumulator[pointer] += 1
#             patterns_left.append(Re_accumulator)
#             patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
#             Re_turn_accumulator.append(pointer)
#             patterns_turn.append(Re_turn_accumulator)
#             return
#         elif Re_stage == n:
#             return
#         else:
#             traverse = bool(Re_end_of_the_stage)
#             cut += traverse
#             paste += (Re_stage - stage + traverse - 1) * (Re_stage - stage)
#             Re_accumulator[pointer] += 1
#             Re_turn_accumulator.append(pointer)
            
#             if Re_end_of_the_stage <= losses1:
#                 patterns_left.append(Re_accumulator)
#                 patterns_right.append([Re_end_of_the_stage, Re_stage + traverse, cut, paste])
#                 patterns_turn.append(Re_turn_accumulator)
#                 pointer = 0
#                 decomposition1(l, L, n, count, cut, paste, Re_accumulator, Re_turn_accumulator, pointer)
#             else:
#                 decomposition1(l, L, n, count, cut, paste, Re_accumulator, Re_turn_accumulator, pointer)

#             # decomposition1(l, L, n, count, cut, paste, Re_accumulator, Re_turn_accumulator, pointer)
            
#     elif Re_stage == n and Re_end_of_the_stage == 0:
#         Re_accumulator[pointer] += 1
#         patterns_left.append(Re_accumulator)
#         patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
#         Re_turn_accumulator.append(pointer)
#         patterns_turn.append(Re_turn_accumulator)
#         return
#     elif Re_stage == n:
#         return
#     else:
#         decomposition1(l, L, n, count - L_values[pointer], cut, paste, Re_accumulator, Re_turn_accumulator, pointer + 1)
#         traverse = bool(Re_end_of_the_stage)
#         cut += traverse
#         paste += (Re_stage - stage + traverse - 1) * (Re_stage - stage)
#         Re_accumulator[pointer] += 1
#         Re_turn_accumulator.append(pointer)
            
#         if Re_end_of_the_stage <= losses1:
#             patterns_left.append(Re_accumulator)
#             patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
#             patterns_turn.append(Re_turn_accumulator)
#             pointer = 0
#             decomposition1(l, L, n, count, cut, paste, Re_accumulator, Re_turn_accumulator, pointer)
#         else:
#             decomposition1(l, L, n, count, cut, paste, Re_accumulator, Re_turn_accumulator, pointer)
# decomposition1(l, L, radius, count, cut, paste, accumulator, turn_accumulator, pointer)
# def patterns_simplify(patterns_left, patterns_right, patterns_turn):
#     patterns_left_plus, patterns_right_plus = [], []
#     patterns_turn_plus = []
#     k = len(patterns_left)
#     for i in range(k):
#         patl = patterns_left[i]
#         if patl not in patterns_left_plus:
#             patterns_left_plus.append(patl)
#             patterns_right_plus.append(patterns_right[i])
#             patterns_turn_plus.append(patterns_turn[i])
#         else:
#             origin_index = patterns_left_plus.index(patl)
#             cut = patterns_right_plus[origin_index][2]
#             if patterns_right[i][2] < cut:
#                 patterns_left_plus.pop(origin_index)
#                 patterns_right_plus.pop(origin_index)
#                 patterns_left_plus.append(patl)
#                 patterns_right_plus.append(patterns_right[i])
#                 patterns_turn_plus.pop(origin_index)
#                 patterns_turn_plus.append(patterns_turn[i])

#     patterns = list(zip(patterns_left_plus, patterns_right_plus))
#     patterns_main, patterns_property, patterns_composition = {}, {}, {}
#     for i in range(len(patterns_left_plus)):
#         patterns_main[i] = [patterns_left_plus[i], patterns_right_plus[i][0]]
#         patterns_property[i] = patterns_right_plus[i]
#         patterns_composition[i] = patterns_turn_plus[i]
#     return patterns, patterns_main, patterns_property, patterns_composition



# def patterns_show(patterns_composition):
#     com = {}
#     k = list(L.values())
#     count = 0
#     for key in patterns_composition:
#         detach = {0: []}
#         composition = patterns_composition[key]
#         length = 0
#         stage = 0
#         for i in composition:   
#             length += k[i]
            
#             if length <= l:
#                 a = detach[stage]
#                 a.append(k[i])
#                 detach[stage] = a

#             else:
#                 length = length - l
#                 a = detach[stage]
#                 a.append(k[i] - length)
#                 detach[stage] = a
#                 stage += 1
#                 detach[stage] = [length]

#         if length < l:
#             a = detach[stage]
#             a.append(l - length)
#             detach[stage] = a

#         com[count] = detach
#         count += 1
#     return com
# patterns, patterns_main, patterns_property, patterns_composition = patterns_simplify(patterns_left, patterns_right, patterns_turn)
# com = patterns_show(patterns_composition)
# print(patterns_composition)
# for i, content in enumerate(patterns):
#     print(i, content)
# # print('patterns:', patterns)
# for key in com:
#     print(key, com[key])

# for i in [10]:
#     p1 = patterns[i]
#     c1 = com[i]
#     print('explaination:')
#     print(f'patterns[{i}]:{p1}')
#     print(f'the number of L1, L2, L3 is: {p1[0][0]}, {p1[0][1]}, {p1[0][2]}')
#     print(f'the left of pattern[{i}] is: {p1[1][0]}')
#     print(f'the number of l is {p1[1][1]}')
#     print(f'need to cut {p1[1][2]} times')
#     print(f'need to paste {p1[1][3]} times')
#     print(f'the composition of pattern[{i}] is: \n {c1}')






























#     # 计算pattern
# def accum2(result):
#         # for calculating how many patterns are used
#     op = []
#         # fake_op用于统计在逼近need时所用的pattern的种类及个数
#     fake_op = copy.deepcopy(result)
#     for key in result:
#         fake_op[key] = 0
#     '''fake_op: {0: 0, 1: 0, 2: 0}'''
#         # convert用于将accum2中尾料更少和先前的pattern的序号一一对应起来
#     convert = {}
#     '''convert: {0: 1, 1: 4, 2: 5, 3: 8}'''
#         # convert中的序列count
#     count = 0
#     for key in result:
#         x, y = result[key]
#             # define the loss you want
#         if y <= losses2:
#                 # 用numpy的array是为了后面计算方便
#             op.append(np.array(x))
#             convert[count] = key
#             count += 1
#     '''op: [array([5, 2, 4]), array([5, 8, 1])]'''
#     print(op)
#         # 方法：比例 + 随机， 目的是用尾料最少的pattern组装或逼近所需的目标数目
#     def ratio(op, need):
#         ratio_need = np.array(need / np.sum(need))
#         possibility_list = np.array([])
#         for i in op:
#                 # 提高各组分比例离目标比例最近的pattern被选中的概率
#                 # pattern的比例
#             ratio_type = np.array(i / np.sum(i))
#                 # pattern比例和目标need比例的距离
#             ratio_distance = np.sum((ratio_type - ratio_need) ** 2)
#             possibility_list = np.append(possibility_list, ratio_distance)
#             '''possibility_list: [0.12809917 0.09141508 0.36264978]'''
#             # 4只是设定的一个参数，可以修改，取4次方只是因为逼近效果好一点
#         m = np.mean(possibility_list) ** 4
#         po1 = 1 / (possibility_list + np.array([m for _ in possibility_list])) ** 4
#             # po1 = 1 / possibility_list ** 4
#             # 将概率调整至和为1的情况
#             # print(po1)
#         po2 = po1 / np.sum(po1)
#             # print(po2)
#         '''po2: [1.31134366e-01 1.09466700e-01 1.11207025e-03 1.31846626e-02
#                     6.22324483e-03 6.02589438e-01 1.45555565e-05 1.50820746e-04
#                     1.31134366e-01 4.11999268e-03 8.69783355e-04]'''
#         return po2
#     curr_need = copy.deepcopy(need)
#     while True:
#             # r = po2
#         r = ratio(op, curr_need)
#             # 随机选择一个pattern进行叠加
#         choose = np.random.choice(range(len(op)), p = r)
#             # 对当前的need进行削减
#         curr_need = curr_need - op[choose]
#             # print(curr_need)
#             # 截止条件：当当前的need出现有小于0的项时，返回前一个need、所用的pattern的种类及个数
#         if all(curr_need == 0):
#             return curr_need, fake_op
#         elif any(curr_need < 0):
#             return curr_need + op[choose], fake_op
#         # 如果没有到截止条件，则计入选中的pattern
#         fake_op[convert[choose]] += 1
# left, acc = accum2(patterns_main)
# '''left: [2 0 1], 
#     acc: {0: 0, 1: 22, 2: 0, 3: 0, 4: 29, 5: 0, 6: 0, 7: 2, 8: 0, 9: 1, 
#           10: 0, 11: 29, 12: 0, 13: 0, 14: 19, 15: 0, 16: 0, 17: 0, 18: 0}'''
# print(left, acc)

#     # accumulator是统计最终用了多少个pattern用的
# accumulator = {}
# for key in patterns_main:
#         accumulator[key] = 0
# '''accumulator: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 
#                 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0}'''
        
        
        
            ########################################################################
#%%
import numpy as np
import copy


# l = 12000
# n = 3
# L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# radius = 5
# losses1 = 0

# def decomposition1(l, L, n, count, cut, paste, accumulator, pointer):
#     L_values = list(L.values())
#     stage = count // l
#     count += L_values[pointer]
#     Re_stage, Re_end_of_the_stage = count // l, count % l
#     Re_end_of_the_stage = l * (Re_stage + bool(Re_end_of_the_stage)) - count
#     Re_accumulator = copy.deepcopy(accumulator)
#     if pointer == len(L_values) - 1:
#         if Re_stage == n and Re_end_of_the_stage == 0:
#             Re_accumulator[pointer] += 1
#             patterns_left.append(Re_accumulator)
#             patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
#             return
#         elif Re_stage == n:
#             return
#         else:
#             # traverse：是否跨越了一个原料长度
#             traverse = bool(Re_end_of_the_stage)
#             cut += traverse
#             paste += (Re_stage - stage + traverse - 1) * (Re_stage - stage)
#             Re_accumulator[pointer] += 1
            
#             if Re_end_of_the_stage <= losses1:
#                 patterns_left.append(Re_accumulator)
#                 patterns_right.append([Re_end_of_the_stage, Re_stage + traverse, cut, paste])
#                 pointer = 0
#                 decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)
#             else:
#                 decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)

#     elif Re_stage == n and Re_end_of_the_stage == 0:
#         Re_accumulator[pointer] += 1
#         patterns_left.append(Re_accumulator)
#         patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
#         return
#     elif Re_stage == n:
#         return
#     else:
#         decomposition1(l, L, n, count - L_values[pointer], cut, paste, Re_accumulator, pointer + 1)
#         traverse = bool(Re_end_of_the_stage)
#         cut += traverse
#         paste += (Re_stage - stage + traverse - 1) * (Re_stage - stage)
#         Re_accumulator[pointer] += 1
            
#         if Re_end_of_the_stage <= losses1:
#             patterns_left.append(Re_accumulator)
#             patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
#             pointer = 0
#             decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)
#         else:
#             decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)

# 不适合L中带有特别大的Li的情况（Li几乎等于原料）
def decomposition2(l, L, n, length, cut, paste, accumulator, path_accumulator, pointer, stage):
    L_length = list(L.values())
    if pointer == len(L_length) - 1:
        if stage == n:
            if len(path_accumulator) >= 2:
                for pattern in path_accumulator[1:]:
                    if pattern not in patterns_path:
                        patterns_path.append(pattern)
            return
        Re_accumulator = copy.deepcopy(accumulator)
        Re_path_accumulator = copy.deepcopy(path_accumulator)
        
        length += L_length[pointer]
        Re_accumulator[pointer] += 1
        if length > l:
            length = length - l
            stage += 1
            cut += 1
            paste += 1
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)
        elif length == l:
            length = 0
            patterns_left.append(Re_accumulator)
            patterns_right.append([0, stage + 1, cut, paste])
            Re_path_accumulator.append(Re_accumulator)
            stage += 1
            pointer = 0
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)
        else:
            cut += 1
            left = l - length
            if left <= losses1:
                patterns_left.append(Re_accumulator)
                patterns_right.append([left, stage + 1, cut, paste])
                Re_path_accumulator.append(Re_accumulator)
                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)
            else:
                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)
    else:
        if stage == n:
            if len(path_accumulator) >= 2:
                for pattern in path_accumulator[1:]:
                    if pattern not in patterns_path:
                        patterns_path.append(pattern)
            return
        Re_accumulator = copy.deepcopy(accumulator)
        Re_path_accumulator = copy.deepcopy(path_accumulator)
        decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer + 1, stage)

        length += L_length[pointer]
        Re_accumulator[pointer] += 1
        if length > l:
            length = length - l
            stage += 1
            cut += 1
            paste += 1
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)
        elif length == l:
            length = 0
            patterns_left.append(Re_accumulator)
            patterns_right.append([0, stage + 1, cut, paste])
            Re_path_accumulator.append(Re_accumulator)
            stage += 1
            pointer = 0
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)
        else:
            cut += 1
            left = l - length
            if left <= losses1:
                patterns_left.append(Re_accumulator)
                patterns_right.append([left, stage + 1, cut, paste])
                Re_path_accumulator.append(Re_accumulator)
                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)
            else:
                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, pointer, stage)


        
def patterns_simplify(patterns_left, patterns_right):
    # 用来缩减重复pattern的数量，同时去除同样的pattern但cut，paste更多的pattern
    patterns_left_plus, patterns_right_plus = [], []
    k = len(patterns_left)
    for i in range(k):
        patl = patterns_left[i]
        if patl not in patterns_left_plus:
            patterns_left_plus.append(patl)
            patterns_right_plus.append(patterns_right[i])
        else:
            origin_index = patterns_left_plus.index(patl)
            cut = patterns_right_plus[origin_index][2]
            if patterns_right[i][2] < cut:
                patterns_left_plus.pop(origin_index)
                patterns_right_plus.pop(origin_index)
                patterns_left_plus.append(patl)
                patterns_right_plus.append(patterns_right[i])
    patterns = list(zip(patterns_left_plus, patterns_right_plus))
    patterns_main, patterns_property = {}, {}
    for i in range(len(patterns_left_plus)):
        # patterns_main[i] = [patterns_left_plus[i], patterns_right_plus[i][0]]
        patterns_main[i] = patterns_left_plus[i]
        patterns_property[i] = patterns_right_plus[i]
    return patterns, patterns_main, patterns_property

def patterns_repeated(patterns, patterns_path):
    p = []
    for i in patterns:
        p.append(i[0])
    for paths in patterns_path:
        loc = p.index(paths)
        patterns.pop(loc)
        p.pop(loc)
    return patterns

def patterns_decomposition(pattern, l, L, joint, length, count, pointer, stage):
# 通过遍历寻找pattern的不同组成方法
# 加入了joint来约束余料长度，顺便剪枝

        p_length = len(pattern)
        L_length = list(L.values())
        Re_pattern = copy.deepcopy(pattern)
        Re_count = copy.deepcopy(count)

        if Re_pattern[pointer] == 0:
            pointer = (pointer + 1) % p_length
        else:
            length += L_length[pointer]
            

            if length > l:
                length -= l
                a = Re_count[stage]

                left = L_length[pointer] - length
                if left < joint or length < joint:
                    return
                
                a.append(left)
                Re_count[stage] = a
                stage += 1
                Re_count[stage] = [length]
                Re_pattern[pointer] = Re_pattern[pointer] - 1

                if not any(Re_pattern):
                    if l == length:
                        Re_count.pop(stage)
                        accumulator.append(Re_count)
                        return
                    else:
                        left = l - length
                        a = Re_count[stage]
                        a.append(left)
                        Re_count[stage] = a
                        accumulator.append(Re_count)
                        return

                                     
                else:
                    for step in range(p_length):
                        Re_pointer = step
                        if Re_pattern[Re_pointer] != 0:
                            patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)

            elif length == l:
                length = 0
                a = Re_count[stage]
                a.append(L_length[pointer])
                Re_count[stage] = a
                stage += 1
                Re_count[stage] = []
                Re_pattern[pointer] = Re_pattern[pointer] - 1

                if not any(Re_pattern):
                    Re_count.pop(stage)
                    accumulator.append(Re_count)
                    return
                    
                else:
                    for step in range(p_length):
                        Re_pointer = step
                        if Re_pattern[Re_pointer] != 0:
                            patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)

            else:
                a = Re_count[stage]
                a.append(L_length[pointer])
                Re_count[stage] = a
                Re_pattern[pointer] = Re_pattern[pointer] - 1

                if not any(Re_pattern):
                    left = l - length
                    a = Re_count[stage]
                    a.append(left)
                    Re_count[stage] = a
                    accumulator.append(Re_count)
                    return
            
                else:
                    for step in range(p_length):
                        Re_pointer = step
                        if Re_pattern[Re_pointer] != 0:
                            patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)
def patterns_decomposition_summon(pattern, l, L, joint):
    p_length = len(pattern)
    for i in range(p_length):
        if pattern[i] != 0:
            patterns_decomposition(pattern, l, L, joint, 0, {0: []}, i, 0)
    return

fill = input('fill in data or by default: press 0 or 1: ')
if fill == '0':
    l = int(input('raw material length: '))
    n = int(input('the number of objects: '))
    L = {}
    for i in range(n):
        L[i] = int(input(f'object L{i + 1}: '))
    radius = int(input('radius of the number of raw materials: '))
    losses1 = int(input('max left of patterns: '))
else:
    l = 12000
    L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
    radius = 10
    losses1 = 0

patterns_left = []
patterns_right = []
patterns_path = []
# cut = 0
# paste = 0
# count = 0
# accumulator = [0 for _ in L.keys()]
# pointer = 0
# decomposition1(l, L, radius, count, cut, paste, accumulator, pointer)
cut = 0
paste = 0
accumulator = [0 for _ in L.keys()]
pointer = 0
stage = 0
length = 0
path_accumulator = []
decomposition2(l, L, radius, length, cut, paste, accumulator, path_accumulator, pointer, stage)
print(patterns_path)

patterns, patterns_main, patterns_property = patterns_simplify(patterns_left, patterns_right)
patterns = patterns_repeated(patterns, patterns_path)
# 会出现pattern1+pattern2=pattern3的情况，虽然说在后续合成过程中不会影响结果，但是会使运算上升一个维度
# 但目前没有很好的处理这样的pattern3的方法

for i, pattern in enumerate(patterns):
    print(i, pattern)


while True:
    check = input('Check the composition of patterns? press enter to keep in / press anything to cancel ').strip().lower()
    if len(check) != 0:
        break
    joint = int(input('min length of joint:(recommended: <20:1000; >20:1500) '))
    patterns_number = len(patterns)
    pattern_index = int(input(f'choose a pattern for its composition: (range:[{0} ~ {patterns_number - 1}]) '))
    pattern = patterns[pattern_index][0]
    accumulator = []
    patterns_decomposition_summon(pattern, l, L, joint)
# 在所有组成的遍历当中，仍然存在问题：对于可以由另外两个pattern1、pattern2组成的pattern3，其分解方式的不同会导致cut和paste的改变
    # for i, a in enumerate(accumulator):
    #     print(i, a)
    if not any(accumulator):
        print('No composition found')
# 计划用sort处理几乎相同的pattern组成（元素交换顺序但不改变余料长度）
# 没办法用set，因为可能出现(1, 2, 2, 3, 3, 4),(1, 1, 2, 3, 4, 4)的情况
# *已解决
    sorted_accumulator = []
    good_list = []
    for i, pattern_sort in enumerate(accumulator):
        new_pattern = {tuple(sorted(pattern_sort[key])) for key in pattern_sort}

        new_pattern_set = new_pattern
        if new_pattern_set not in sorted_accumulator:
            sorted_accumulator.append(new_pattern_set)
            good_list.append(i)

    smaller_accumulator = []
    for i in good_list:
        smaller_accumulator.append(accumulator[i])
    for i, pattern in enumerate(smaller_accumulator):
        print(i, pattern)    