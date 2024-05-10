#################
import numpy as np
import copy
l = 12000
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
need = np.array([852, 658, 162])
radius = 9
losses1 = 60
losses2 = 40

'''patterns: [([5, 2, 4], [0, 4, 10, 3]), ([number1, number2, number3], [left, ls, cut, paste])]
cut, paste: integers
count: length
accumulator: [5, 2, 4] ~ the number of L1 L2 L3

'''
patterns_left = []
patterns_right = []
cut = 0
paste = 0
count = 0
accumulator = [0 for _ in L.keys()]
pointer = 0
def decomposition1(l, L, n, count, cut, paste, accumulator, pointer):
    L_values = list(L.values())
    stage = count // l
    count += L_values[pointer]
    Re_stage, Re_end_of_the_stage = count // l, count % l
    Re_end_of_the_stage = l * (Re_stage + bool(Re_end_of_the_stage)) - count
    Re_accumulator = copy.deepcopy(accumulator)
    if pointer == len(L_values) - 1:
        if Re_stage == n and Re_end_of_the_stage == 0:
            Re_accumulator[pointer] += 1
            patterns_left.append(Re_accumulator)
            patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
            return
        elif Re_stage == n:
            return
        else:
            traverse = bool(Re_end_of_the_stage)
            cut += traverse
            paste += (Re_stage - stage + traverse - 1) * (Re_stage - stage)
            Re_accumulator[pointer] += 1
            if Re_end_of_the_stage <= losses1:
                patterns_left.append(Re_accumulator)
                patterns_right.append([Re_end_of_the_stage, Re_stage + traverse, cut, paste])
                pointer = 0
                decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)
            else:
                decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)
    elif Re_stage == n and Re_end_of_the_stage == 0:
        Re_accumulator[pointer] += 1
        patterns_left.append(Re_accumulator)
        patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
        return
    elif Re_stage == n:
        return
    else:
        decomposition1(l, L, n, count - L_values[pointer], cut, paste, Re_accumulator, pointer + 1)
        traverse = bool(Re_end_of_the_stage)
        cut += traverse
        paste += (Re_stage - stage + traverse - 1) * (Re_stage - stage)
        Re_accumulator[pointer] += 1
        if Re_end_of_the_stage <= losses1:
            patterns_left.append(Re_accumulator)
            patterns_right.append([Re_end_of_the_stage, Re_stage, cut, paste])
            pointer = 0
            decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)
        else:
            decomposition1(l, L, n, count, cut, paste, Re_accumulator, pointer)
decomposition1(l, L, radius, count, cut, paste, accumulator, pointer)
def patterns_simplify(patterns_left, patterns_right):
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
        patterns_main[i] = [patterns_left_plus[i], patterns_right_plus[i][0]]
        patterns_property[i] = patterns_right_plus[i]
    return patterns, patterns_main, patterns_property
patterns, patterns_main, patterns_property = patterns_simplify(patterns_left, patterns_right)
print(patterns, patterns_main, patterns_property, sep = '\n')

    # 计算pattern
def accum2(result):
        # for calculating how many patterns are used
    op = []
        # fake_op用于统计在逼近need时所用的pattern的种类及个数
    fake_op = copy.deepcopy(result)
    for key in result:
        fake_op[key] = 0
    '''fake_op: {0: 0, 1: 0, 2: 0}'''
        # convert用于将accum2中尾料更少和先前的pattern的序号一一对应起来
    convert = {}
    '''convert: {0: 1, 1: 4, 2: 5, 3: 8}'''
        # convert中的序列count
    count = 0
    for key in result:
        x, y = result[key]
            # define the loss you want
        if y <= losses2:
                # 用numpy的array是为了后面计算方便
            op.append(np.array(x))
            convert[count] = key
            count += 1
    '''op: [array([5, 2, 4]), array([5, 8, 1])]'''
    print(op)
        # 方法：比例 + 随机， 目的是用尾料最少的pattern组装或逼近所需的目标数目
    def ratio(op, need):
        ratio_need = np.array(need / np.sum(need))
        possibility_list = np.array([])
        for i in op:
                # 提高各组分比例离目标比例最近的pattern被选中的概率
                # pattern的比例
            ratio_type = np.array(i / np.sum(i))
                # pattern比例和目标need比例的距离
            ratio_distance = np.sum((ratio_type - ratio_need) ** 2)
            possibility_list = np.append(possibility_list, ratio_distance)
            '''possibility_list: [0.12809917 0.09141508 0.36264978]'''
            # 4只是设定的一个参数，可以修改，取4次方只是因为逼近效果好一点
        m = np.mean(possibility_list) ** 4
        po1 = 1 / (possibility_list + np.array([m for _ in possibility_list])) ** 4
            # po1 = 1 / possibility_list ** 4
            # 将概率调整至和为1的情况
            # print(po1)
        po2 = po1 / np.sum(po1)
            # print(po2)
        '''po2: [1.31134366e-01 1.09466700e-01 1.11207025e-03 1.31846626e-02
                    6.22324483e-03 6.02589438e-01 1.45555565e-05 1.50820746e-04
                    1.31134366e-01 4.11999268e-03 8.69783355e-04]'''
        return po2
    curr_need = copy.deepcopy(need)
    while True:
            # r = po2
        r = ratio(op, curr_need)
            # 随机选择一个pattern进行叠加
        choose = np.random.choice(range(len(op)), p = r)
            # 对当前的need进行削减
        curr_need = curr_need - op[choose]
            # print(curr_need)
            # 截止条件：当当前的need出现有小于0的项时，返回前一个need、所用的pattern的种类及个数
        if all(curr_need == 0):
            return curr_need, fake_op
        elif any(curr_need < 0):
            return curr_need + op[choose], fake_op
        # 如果没有到截止条件，则计入选中的pattern
        fake_op[convert[choose]] += 1
left, acc = accum2(patterns_main)
'''left: [2 0 1], 
    acc: {0: 0, 1: 22, 2: 0, 3: 0, 4: 29, 5: 0, 6: 0, 7: 2, 8: 0, 9: 1, 
          10: 0, 11: 29, 12: 0, 13: 0, 14: 19, 15: 0, 16: 0, 17: 0, 18: 0}'''
print(left, acc)

    # accumulator是统计最终用了多少个pattern用的
accumulator = {}
for key in patterns_main:
        accumulator[key] = 0
'''accumulator: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 
                10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0}'''
        
        
        
            
        