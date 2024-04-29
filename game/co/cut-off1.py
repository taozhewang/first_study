#%%
import numpy as np
from core import pattern_oringin, calc_loss_joint, calc_cost
import copy

# 原始钢筋长度
l = 12000
# 钢筋的规格
l_size = 32
# 目标钢筋长度
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# 目标钢筋的数量
need = np.array([552, 658, 462])

# 最大的组合数
radius = 10

# 组合数最小余料
losses1 = 50

'''
按最小余料，找出最长 radius 的组合，并计算其余料
返回值：
idx: 组合的id: [pattern: {} 组合的种类:数量, loss: 组合的余料, joint: 接头的数量, cost: 成本, eer: 能效比=cost/nums]
patterns:{idx: [pattern, loss, joint, cost, eer]} 
'''
def decom(l, L):
    
    # 求各种组合的列表
    patterns = pattern_oringin(l, L, losses1, radius)
    '''patterns: {0: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 0,0,0,0],
                1: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 50,3,400,100]}'''
    patterns_length = len(patterns)
    print(f"patterns[0]:", patterns[0])
    print(f"patterns[{patterns_length}]:", patterns[patterns_length-1])
    print(f"patterns length: {patterns_length}")

    # 求组合的的使用情况
    def accum2(patterns):
        # for calculating how many patterns are used
        op = []
        # fake_op用于统计在逼近need时所用的pattern的种类及个数
        fake_op = {}
        for key in patterns:
            fake_op[key] = 0

        # op id 转换为 pattern id
        op_2_pattern = []
        # 计算50%能效比分位数
        eer_percentile  = np.percentile([patterns[key][4] for key in patterns],50)
        for key in patterns:
            pattern, loss, joint, cost, eer = patterns[key]   

            # 这里按照小于等于能效比来选择pattern：
            if eer <= eer_percentile:
                # 用numpy的array是为了后面计算方便
                op.append(np.array(list(pattern.values())))
                op_2_pattern.append(key)
        '''
            fake_op: {0: 0, 1: 0, 2: 0} 
            op: [array([5, 2, 4]), array([5, 8, 1])]
            convert: {0: 1, 1: 4, 2: 5, 3: 8}
        '''
        assert len(op) > 0, "no pattern can be used"
        print("op length:", len(op))
        
        # 方法：比例 + 随机， 目的是用尾料最少的pattern组装或逼近所需的目标数目
        def ratio(op, need):
            #  目标占比
            ratio_need = np.array(need / np.sum(need))
            
            # 所有组合和目标比例的欧式距离
            # possibility_list: [0.12809917 0.09141508 0.36264978]
            possibility_list = np.array([])
            for i in op:
                # 提高各组分比例离目标比例最近的pattern被选中的概率
                # pattern的比例
                ratio_type = np.array(i / np.sum(i))
                
                # pattern比例和目标need比例的欧式距离
                ratio_distance = np.sum((ratio_type - ratio_need) ** 2)
                
                possibility_list = np.append(possibility_list, ratio_distance)
                
            # 相似度为距离的反比，取4次方只是因为逼近效果好一点
            po1 = 1 / possibility_list ** 4
            # 将相似度归一化为概率
            po2 = po1 / np.sum(po1)
            '''po2: [1.31134366e-01 1.09466700e-01 1.11207025e-03 1.31846626e-02
                    6.22324483e-03 6.02589438e-01 1.45555565e-05 1.50820746e-04
                    1.31134366e-01 4.11999268e-03 8.69783355e-04]'''
            return po2
        
        curr_need = copy.deepcopy(need) # 需要的目标数目
        op_idxs = list(range(len(op)))  # 所有op的id   
        while True:
            # 按 组合比例 和 剩余目标比例 的相似度 求 采样概率
            r = ratio(op, curr_need)
            
            # 随机选择一个pattern进行叠加
            choose = np.random.choice(op_idxs, p = r)
            
            # 对当前的need进行削减
            curr_need = curr_need - op[choose]
            
            # 截止条件：当当前的need出现有小于0的项时，返回前一个need、所用的pattern的种类及个数
            if any(curr_need < 0):                
                return curr_need + op[choose], fake_op
            
            # 如果没有到截止条件，则计入选中的pattern
            fake_op[op_2_pattern[choose]] += 1
    
    # 剩余匹配量，当前patterns的使用量            
    left, acc = accum2(patterns)
    '''left: [2 0 1], 
    acc: {0: 0, 1: 22, 2: 0, 3: 0, 4: 29, 5: 0, 6: 0, 7: 2, 8: 0, 9: 1, 
          10: 0, 11: 29, 12: 0, 13: 0, 14: 19, 15: 0, 16: 0, 17: 0, 18: 0}'''

    print("第一次匹配：")
    print([(key, value) for key, value in acc.items() if value > 0])
    print("剩余：")
    print(left)

    # 第二次匹配，用剩余的 left 来组合
    # 此处用尾料没那么少的pattern来组合
    # 在need降低到足够小的时候，使用遍历找出满足need的组合
    # (还有一种方法，直接将need剩余的项从原料中取，而不依靠pattern)
    # 组合的方法集合ways
    ways = {}    
    '''
        patterns: 组合种类
        left: 剩余情况
        accumulator: 使用方案
        pointer: 当前组合id
        count: 输出组合计数id
    '''
    # 将所有的递归改为循环，减少递归深度
    def accum3(patterns, left):
        '''patterns: {0: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 0],
                    1: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 50]}'''
        accumulator={}
        pointer=0
        count=0 
        stack = [(left, accumulator, pointer, count)]
        i = 0
        while stack:
            left, accumulator, pointer, count = stack.pop()   

            i += 1
            if i % 100000 == 0:
                print(f"stack length: {len(stack)}  ways length: {len(ways)}  left: {left}")

            # 当剩余数量全部数量全部都为0时才返回组合
            if np.all(left == 0):
                # 将当前使用方案储存进ways里面，返回
                ways[count] = accumulator
                # 限制ways的长度为5，防止内存溢出
                if len(ways)>=5: return
                continue

            # 如果pointer已经指向最后一个pattern，则跳过
            if pointer >= len(patterns):
                continue

            # 取得当前 pointer 指向的 pattern
            pattern, loss, joint, cost, eer = patterns[pointer]        
            pattern_values = np.array(list(pattern.values()))

            # 扣减数量
            l = left - pattern_values
            # 如果有负数，则跳过
            if not np.any(l < 0): 
                ac = copy.deepcopy(accumulator)
                if pointer in ac:
                    ac[pointer] += 1
                else:
                    ac[pointer] = 1
                stack.append((l, ac, 0, count+1))

            stack.append((left, accumulator, pointer+1, count))  

    accum3(patterns, left)
    print(ways)

    # 统计出所有目前计算得到的组合，并将其用料和尾料显示出来
    def find_min1():
        more = []
        combination = []
        for way in ways.values():

            # 合并相同的pattern的数量    
            total_sum = {}
            for key in acc:
                if key in way:
                    total_sum[key] = way[key] + acc[key]
                else:
                    total_sum[key] = acc[key]

                '''(for example) 
                total_sum: {0: 0, 1: 18, 2: 0, 3: 0, 4: 27, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1, 
                            10: 0, 11: 31, 12: 0, 13: 0, 14: 22, 15: 0, 16: 0, 17: 0, 18: 2}'''
            materials = np.array([0 for _ in range(len(need))])
            '''materials: [0, 0, 0]'''
            loss =0
            joint=0
            cost =0
            for key in total_sum:
                pattern = patterns[key][0]
                '''pattern: {'L1': xx, 'L2': xx, 'L3': xx}'''
                pattern_values = list(pattern.values())
                '''pattern_values: [xx, xx, xx]'''
                materials += np.array(pattern_values) * total_sum[key]
                loss += patterns[key][1] * total_sum[key]
                joint += patterns[key][2] * total_sum[key]
                cost += patterns[key][3] * total_sum[key]
       
            combination.append(materials)
            more.append([loss, joint, cost])
        return more ,combination
    
    x, y= find_min1()
    # 打印得到的结果
    statistics = list(zip(y, x))
    count = 0
    for i in statistics:
        print(f'{count + 1}: combination: \n {y[count]} \n loss: {i[1][0]} \n joint: {i[1][1]} \n cost: {i[1][2]}')
        count += 1
        
        
decom(l, L)