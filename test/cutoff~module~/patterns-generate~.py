import numpy as np
import copy
import time

# 下面很可能是产生组合的什么东西
def patterns_generate(l, L, joint, radius, losses, length, accumulator, stage, waste, paste, waste_cost, paste_cost, order):
    patterns = [] # 用于记录所有形如[12, 4, 5]的组合 
    patterns_property = [] # 用于记录所有组合的成本、尾料、粘合次数
    patterns_order = [] # 用于记录所有组合的物料顺序
    group = [] # 用于记录所有种类的组合，在成本一样的情况下不考虑其排序
    pat = [] # 用于记录所有种类的组合的字符串形式

    stack = [(length, accumulator, stage, waste, paste, pointer, order) for pointer in range(len(L))]

    while stack:
        # print(len(stack))
        # length, accumulator, stage, waste, paste, pointer, order = stack.pop(0) # 优先广度搜索
        length, accumulator, stage, waste, paste, pointer, order = stack.pop() # 优先深度搜索
        
        if stage == radius: # 如果到达最大原料用量，那么前面需要停止
            continue

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
                    continue
                pat.append(p) # 登记组合

                patterns.append(caccumulator)
                patterns_property.append([cost, cwaste, cpaste])
                patterns_order.append(corder)
            continue

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
                    continue
                pat.append(p)

                patterns.append(caccumulator)
                patterns_property.append([cost, cwaste, cpaste])
                patterns_order.append(corder)
                
            clength = 0
            cstage += 1

        else:
            # print(corder) # 测试用选项
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
                    continue
                pat.append(p)

                patterns.append(caccumulator)
                patterns_property.append([cost, ccwaste, cpaste])
                patterns_order.append(corder)

        present = '_'.join([str(x) for x in caccumulator] + [str(cwaste), str(cpaste)]) # 用于登记当前状态：组合+尾料+粘合次数
        if present in group:                                                            # 如果该状态已经被登记过，则跳过，用于剪枝减少计算量
            continue
        group.append(present)

        for pointer in range(len(L)):
            stack.append((clength, caccumulator, cstage, cwaste, cpaste, pointer, corder))

    return patterns, patterns_property, patterns_order

l = 12000
L = np.array([4100, 4350, 4700])
joint = 200
radius = 10
losses = 100
length = 0
accumulator = np.zeros(len(L))
stage = 0
waste = 0
paste = 0
waste_cost = 1
paste_cost = 1
order = [[]]

starttime = time.time()
patterns, patterns_property, patterns_order = patterns_generate(l, L, joint, radius, losses, length, accumulator, stage, waste, paste, waste_cost, paste_cost, order)
print(f'patterns: {patterns}')
print(f'patterns_property: {patterns_property}')
print(f'patterns_order: {patterns_order}')
endtime = time.time()
print(f'time: {endtime - starttime}s')