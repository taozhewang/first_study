# #%%
# # cutoff
# import numpy as np
# import copy
# # L 为目标，type 为 dict
# # l 为原料
# # attention: L 必须升序排列，不然第三部分会出错（代码是默认升序写的）
# L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# l = 12000
# need = np.array([[852, 660, 162]])

# def decom(l, L):
#     # 第一步
#     def count_decomposition1(l, L, count, pointer):
#         L_keys = list(L.keys())
#         L_values = list(L.values())
#         if len(L_keys) == pointer:
#             return [count]
#         elif l < L_values[pointer]:
#             return count_decomposition1(l, L, count, pointer + 1)
#         else:
#             return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
#                                         pointer) + count_decomposition1(l, L, count, pointer + 1)
#     result1 = count_decomposition1(l, L, [], 0)
#     # 防止出现[]的情况
#     del result1[len(result1) - 1]
#     print(result1)

#     # 第二步
#     def count_decomposition2(l, L, count, pointer):
#         L_keys = list(L.keys())
#         L_values = list(L.values())
#         if l < 0:
#             return [count]
#         elif pointer == len(L_keys) - 1:
#             return count_decomposition2(l - L_values[pointer], L, count + [L_keys[pointer]], pointer)
#         else:
#             return count_decomposition2(l - L_values[pointer], L, count + [L_keys[pointer]],
#                                         pointer) + count_decomposition2(l, L, count, pointer + 1)
#     result2 = count_decomposition2(l, L, [], 0)
#     print(result2)

#     # 第三步
#     result3_medium = count_decomposition1(2 * l, L, [], 0)
#     # 防止出现不能由尾料减尾尾料得到的组合
#     l1 = min(len(l) for l in result1)
#     l2 = min(len(l) for l in result2)
#     l3 = l1 + l2
#     result3 = []
#     for i in result3_medium:
#         if len(i) >= l3 and i not in result2:
#             result3.append(i)
#     print(result3)
#     return result1, result2, result3
# # decom(l, L)

# # 返回：所有可能出现的组合
# #%%
# l = 12000
# L = {'L1' : 4170, 'L2' : 4350, 'L3' : 4700}

# def decom_quant(l, L):
#     ways = {}
#     def decom(l, L):
#         def count_decomposition1(l, L, count, pointer):
#             L_keys = list(L.keys())
#             L_values = list(L.values())
#             if len(L_keys) == pointer:
#                 return [count]
#             elif l < L_values[pointer]:
#                 return count_decomposition1(l, L, count, pointer + 1)
#             else:
#                 return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
#                                             pointer) + count_decomposition1(l, L, count, pointer + 1)
#         result1 = count_decomposition1(l, L, [], 0)
#         del result1[len(result1) - 1]
#         # print(result1)
#         return result1
#     def quant(l, L, a):
#         for j in a:
#             l = l - L[j]
#         return l
#     result = decom(l, L)
#     count = 0
#     for i in result:
#         count += 1
#         ways[count] = [i, quant(l, L, i)]
#     return ways

# w = decom_quant(4 * l, L)
# t = 0
# k = {}
# for i in w:
#     if w[i][1] < 500:
#         t += 1
#         k[t] = w[i]
# # print(w)
# print(k)
#%%
# p = 11
# def decom_quant(l, L):
#     ways = {}
#     def decom(l, L):
#         def count_decomposition1(l, L, count, pointer):
#             L_keys = list(L.keys())
#             L_values = list(L.values())
#             if len(L_keys) == pointer:
#                 return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
#                                             pointer)
#             elif l < L_values[pointer]:
#                 return [count]
#             else:
#                 return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
#                                             pointer) + count_decomposition1(l, L, count, pointer + 1)
#         result1 = count_decomposition1(l, L, [], 0)
#         print(result1)
#         return result1
 
#     def quant(l, L, a):
#         for j in a:
#             l = l - L[j]
#         return l
#     result = decom(l, L)
#     count = 0
#     for i in result:
#         count += 1
#         ways[count] = [i, quant(l, L, i)]
#     return ways

#%%
# def decom_pattern(l, L):
#     def count_decomposition1(l, L, count, pointer):
#         L_keys = list(L.keys())
#         L_values = list(L.values())
#         if l < 100:
#             return [count]
#         elif l < L_values[pointer]:
#             return []
#         elif len(L_keys) == pointer + 1:
#             return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
#                                         pointer)
#         else:
#             return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
#                                         pointer) + count_decomposition1(l, L, count, pointer + 1)
#     result1 = count_decomposition1(l, L, [], 0)
#     # string -> numbers
#     def scan_result(L, a):
#         Num = []
#         for i in L.keys():
#             num = 0
#             for j in a:
#                 if i == j:
#                     num += 1
#             Num.append(num)
#         return Num
#     # length calculator
#     def quant(l, L, a):
#         for j in a:
#             l = l - L[j]
#         return l
#     # check the length of the left
#     good_pattern = {}
#     count = 0
#     for i in result1:
#         count += 1
#         k = quant(l, L, i)
#         good_pattern[count] = [i, k]
#     # use numbers instead of strings
#     for p in good_pattern.keys():
#         good_pattern[p] = [scan_result(L, good_pattern[p][0]), good_pattern[p][1]]
#     return good_pattern
# count = 0
# result = {}
# for i in range(1, 7):
#     k = decom_pattern(i * l, L)
#     for r in k.values():
#         result[count] = r
#         count += 1
# print(result)
'''
{0: [[1, 3, 4], 50], 1: [[5, 2, 4], 0], 
2: [[1, 9, 1], 50], 3: [[8, 3, 3], 50], 
4: [[5, 8, 1], 0], 5: [[12, 2, 3], 0], 
6: [[8, 9, 0], 50], 7: [[3, 4, 9], 0]}
'''
# R = np.array([i[0] for i in result.values()])
# R = R.T
# print(R)
# R_pseudo_inverse = np.linalg.pinv(R)
# ww = R_pseudo_inverse @ need.T
# print(ww)

# K = {}
# for i in result.keys():
#     K[i] = np.array(result[i][0])
'''
K = {0: [1, 3, 4], 1: [5, 2, 4], 2: [1, 9, 1], 
     3: [8, 3, 3], 4: [5, 8, 1], 5: [12, 2, 3], 
     6: [8, 9, 0], 7: [3, 4, 9]}
'''
# to the sake of your computer, please do not let numbers in 'need' bigger than 10
# need = np.array([4, 2, 3])
# def count_decomposition2(need, K, count, pointer):
#     K_keys = list(K.keys())
#     K_values = list(K.values())
#     low = np.where(need > 0)
#     # print(pointer)
#     minus = np.where(K_values[pointer] == 0)
#     # print(need)
#     if np.array_equal(low, minus) and len(K_keys) > pointer + 1:
#         return count_decomposition2(need, K, count, pointer + 1)
#     elif all(need <= 0):
#         return [count]
#     elif len(K_keys) == pointer + 1:
#         return count_decomposition2(need - K_values[pointer], K, count + [K_keys[pointer]],
#                                     pointer)
#     else:
#         return count_decomposition2(need - K_values[pointer], K, count + [K_keys[pointer]],
#                                     pointer) + count_decomposition2(need, K, count, pointer + 1)
# # rresult = count_decomposition2(need, K, [], 0)
# # print(rresult)
# def find_min(L, need, result, a):
#     # l = np.array([list(L.values())])
#     # # print(l, need)
#     # total = l @ need.T
#     # # print(total)
#     length = []
#     for i in a:
#         print(i)
#         length.append(sum(result[j][1] for j in i))
#         print(length)
    
# # rrresult = find_min(L, need, result, rresult)
# # print(rrresult)
#%%
import numpy as np
import copy
l = 12000
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
need = np.array([852, 658, 162])
losses1 = 50
losses2 = 40
def f():
    def decom_pattern(l, L):
        def count_decomposition1(l, L, count, pointer):
            L_keys = list(L.keys())
            L_values = list(L.values())
            # limit the losses
            if l <= losses1:
                return [count]
            elif l < L_values[pointer]:
                return []
            elif len(L_keys) == pointer + 1:
                return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                            pointer)
            else:
                return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                            pointer) + count_decomposition1(l, L, count, pointer + 1)
        result1 = count_decomposition1(l, L, [], 0)
        # string -> numbers
        def scan_result(L, a):
            Num = []
            for i in L.keys():
                num = 0
                for j in a:
                    if i == j:
                        num += 1
                Num.append(num)
            return Num
        # length calculator
        def quant(l, L, a):
            for j in a:
                l = l - L[j]
            return l
        # check the length of the left
        good_pattern = {}
        count = 0
        for i in result1:
            count += 1
            k = quant(l, L, i)
            good_pattern[count] = [i, k]
        # use numbers instead of strings
        for p in good_pattern.keys():
            good_pattern[p] = [scan_result(L, good_pattern[p][0]), good_pattern[p][1]]
        return good_pattern
    count = 0
    result = {}
    # here we can change the range
    for i in range(1, 11):
        k = decom_pattern(i * l, L)
        for r in k.values():
            result[count] = r
            count += 1
    print(result)
    '''
    {0: [[1, 3, 4], 50], 1: [[5, 2, 4], 0], 
    2: [[1, 9, 1], 50], 3: [[8, 3, 3], 50], 
    4: [[5, 8, 1], 0], 5: [[12, 2, 3], 0], 
    6: [[8, 9, 0], 50], 7: [[3, 4, 9], 0]}
    '''

    def accum2():
        # for calculating how many patterns are used
        op = []
        fake_op = copy.deepcopy(result)
        for i in result.keys():
            fake_op[i] = 0
        convert = {}
        count = 0
        for i in result.keys():
            x, y = result[i]
            # define the loss you want
            if y <= losses2:
                op.append(x)
                convert[count] = i
                count += 1
        print(op)

        def ratio(op, need):
            ratio_need = np.array(need / np.sum(need))
            possibility_list = np.array([])
            for i in op:
                ratio_type = np.array(i / np.sum(i))
                ratio_distance = np.sum((ratio_type - ratio_need) ** 2)
                possibility_list = np.append(possibility_list, ratio_distance)
            po1 = 1 / possibility_list ** 4
            po2 = po1 / np.sum(po1)
            return po2
        curr_need = copy.deepcopy(need)
        while True:
            r = ratio(op, curr_need)
            choose = np.random.choice(range(len(op)), p = r)
            curr_need = curr_need - op[choose]
            if any(curr_need < 0):
                return curr_need + op[choose], fake_op
            fake_op[convert[choose]] += 1
    left, acc = accum2()
    print(left, acc)

    accumulator = {}
    for i in result.keys():
        accumulator[i] = 0

    ways = {}
    def accum3(result, left, accumulator, pointer, countor):
        null_space = np.where(left > 0)
        null_number = np.array([result[pointer][0][i] for i in null_space[0]])
        if all(left <= 0):
            ways[countor] = accumulator
            return accumulator
        elif all(null_number == 0):
            if pointer < len(result.keys()) - 1:
                accum3(result, left, accumulator, pointer + 1, countor)
                return None
            else:
                return None
        elif pointer == len(result.keys()) - 1:
            ac = copy.deepcopy(accumulator)
            ac[pointer] += 1
            l = left - result[pointer][0]
            accum3(result, l, ac, pointer, countor)
            return None
        else:
            ac = copy.deepcopy(accumulator)
            ac[pointer] += 1
            l = left - result[pointer][0]
            accum3(result, l, ac, pointer, countor)
            accum3(result, left, accumulator, pointer + 1, countor + 1)
            None
    accum3(result, left, accumulator, 0, 0)
    print(ways)
    def find_min1():
        more = []
        for way in ways.values():
            total_sum = {}
            for i in ways.keys():
                total_sum[i] = way[i] + acc[i]
            materials = np.array([0 for _ in range(len(need))])
            waste = 0
            for i in total_sum.keys():
                materials += np.array(result[i][0]) * total_sum[i]
                waste += result[i][1] * total_sum[i]
            over_produce = materials - need
            L_length = list(L.values())
            over_p = 0
            for i in range(len(over_produce)):
                over_p += L_length[i] * over_produce[i]
            more.append(np.array([over_p, waste]))
        return more
    m = find_min1()
    print(m)
# f()
#%%
# def decom(l, L):
# # 第一步：从L中允许重复取样，只要满足L的总size小于l的组合   
#     result1 = set()
#     def count_decomposition1(dst_dict, temp_sum, temp_combo, max_sum, result):
#     # 如果当前和大于 max_sum,则剪枝
#         if temp_sum > max_sum:
#             return
#         # 如果当前和大于0,则将当前组合添加到结果中,并去重
#         if temp_sum>0:
#             result.add(tuple(sorted(temp_combo)))
#         # 回溯
#         for key in L:
#             temp_combo.append(key)
#             count_decomposition1(dst_dict, temp_sum + L[key], temp_combo, max_sum, result)
#             temp_combo.pop()
#     count_decomposition1(L, 0, [], l, result1)
#     print(result1)

#     result2 = set()
#     def count_decomposition2(dst_dict, temp_sum, temp_combo, max_sum, result):
#     # 如果当前和大于 max_sum,则加入并剪枝
#         if temp_sum > max_sum:
#             result.add(tuple(sorted(temp_combo)))
#             return
#         # 回溯
#         for key in L:
#             temp_combo.append(key)
#             count_decomposition2(dst_dict, temp_sum + L[key], temp_combo, max_sum, result)
#             temp_combo.pop()
#     count_decomposition2(L, 0, [], l, result2)
#     print(result2)


#     # 第三步：从L中允许重复取样，找出所有满足L的总size小于2*l的组合
#     result3_medium = set()
#     count_decomposition1(L, 0, [], 2*l, result3_medium)
#     print(result3_medium)

#     # 防止出现不能由尾料减尾尾料得到的组合
#     result3 = set()
#     for r in result3_medium:
#         if r not in result1 and r not in result2:
#             result3.add(r)
#     print(result3)
#     return result1, result2, result3
# decom(l, L)

#%%
import numpy as np
import copy
l = 12000
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
need = np.array([552, 658, 462])
radius = 9
losses1 = 50
losses2 = 40
def decom(l, L):
    # 计算pattern
    def count_decomposition1(l, L, count, pointer):
            L_keys = list(L.keys())
            L_values = list(L.values())
            # 只选取尾料在losses1以内的pattern
            if l <= losses1:
                return [count]
            elif l < L_values[pointer]:
                return []
            elif len(L_keys) == pointer + 1:
                return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                            pointer)
            else:
                return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                            pointer) + count_decomposition1(l, L, count, pointer + 1)
    

    # 产生patterns(将n * l分解为Li之和的组合)的函数
    def pattern_oringin(l, L, losses1):

        # 将result的元素从string转到int
        def result_transform(results):
            # 创造一个储存pattern的集合
            result_set = []
            # 创造一个用于储存一个pattern中各种原料Li数量的集合
            result_copy = copy.deepcopy(L)
            '''result_set: [{}, {}, {}]
            result_copy: {'L1' : 0, 'L2' : 0, 'L3' : 0}'''
            for i in L:
                result_copy[i] = 0
            for one_result in results:
                result_num = copy.deepcopy(result_copy)
                for the_kind_of_l in one_result:
                    result_num[the_kind_of_l] += 1
                result_set.append(result_num)
            return result_set
        
        # 计算出每个pattern的尾料
        def pattern_process(result_set, l):
            # good pattern 的个数count
            count = 0
            pattern = {}
            for sets in result_set:
                ''' sets: {'L1' : xx, 'L2' : xx, 'L3' : xx}'''
                accumulate = 0
                for key in sets:
                    # 计算每个pattern的总长度
                    accumulate += sets[key] * L[key]
                # 计算余料left
                left = l - accumulate
                '''pattern: {0: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 0],
                    1: [{'L1' : xx, 'L2' : xx, 'L3' : xx}, 50]}'''
                pattern[count] = [sets, left]
                count += 1
            return pattern
        
        # 产生包含很多pattern的集合patterns
        patterns = {}
        # paterns内元素的个数
        count = 0
        # 合并pattern，组装成patterns
        for times in range(1, radius):
            result1 = set()
            result1 = count_decomposition1(times * l, L, [], 0)
            result_set = result_transform(result1)
            # print(result_set)
            pattern = pattern_process(result_set, times * l)
            # print(pattern)
            for key in pattern:
                patterns[count] = pattern[key]
                count += 1
            # print(patterns)
        return patterns
    patterns = pattern_oringin(l, L, losses1)
    # print(patterns)

    def accum2(result):
        # for calculating how many patterns are used
        op = []
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
                op.append(np.array(list(x.values())))
                convert[count] = key
                count += 1
        '''op: [array([5, 2, 4]), array([5, 8, 1])]'''
        print(op)
        # 方法：比例 + 随机， 目的是用尾料最少的pattern组装或逼近所需的目标数目
        def ratio(op, need):
            ratio_need = np.array(need / np.sum(need))
            possibility_list = np.array([])
            for i in op:
                ratio_type = np.array(i / np.sum(i))
                ratio_distance = np.sum((ratio_type - ratio_need) ** 2)
                possibility_list = np.append(possibility_list, ratio_distance)
            po1 = 1 / possibility_list ** 4
            po2 = po1 / np.sum(po1)
            return po2
        curr_need = copy.deepcopy(need)
        while True:
            r = ratio(op, curr_need)
            choose = np.random.choice(range(len(op)), p = r)
            curr_need = curr_need - op[choose]
            if any(curr_need < 0):
                return curr_need + op[choose], fake_op
            fake_op[convert[choose]] += 1
    left, acc = accum2(patterns)
    print(left, acc)

    accumulator = {}
    for i in patterns.keys():
        accumulator[i] = 0

    ways = {}
    def accum3(result, left, accumulator, pointer, countor):
        null_space = np.where(left > 0)
        null_number = np.array([result[pointer][0][i] for i in null_space[0]])
        if all(left <= 0):
            ways[countor] = accumulator
            return accumulator
        elif all(null_number == 0):
            if pointer < len(result.keys()) - 1:
                accum3(result, left, accumulator, pointer + 1, countor)
                return None
            else:
                return None
        elif pointer == len(result.keys()) - 1:
            ac = copy.deepcopy(accumulator)
            ac[pointer] += 1
            l = left - result[pointer][0]
            accum3(result, l, ac, pointer, countor)
            return None
        else:
            ac = copy.deepcopy(accumulator)
            ac[pointer] += 1
            l = left - result[pointer][0]
            accum3(result, l, ac, pointer, countor)
            accum3(result, left, accumulator, pointer + 1, countor + 1)
            None
    accum3(patterns, left, accumulator, 0, 0)
    print(ways)
    def find_min1():
        more = []
        for way in ways.values():
            total_sum = {}
            for i in ways.keys():
                total_sum[i] = way[i] + acc[i]
            materials = np.array([0 for _ in range(len(need))])
            waste = 0
            for i in total_sum.keys():
                materials += np.array(patterns[i][0]) * total_sum[i]
                waste += patterns[i][1] * total_sum[i]
            over_produce = materials - need
            L_length = list(L.values())
            over_p = 0
            for i in range(len(over_produce)):
                over_p += L_length[i] * over_produce[i]
            more.append(np.array([over_p, waste]))
        return more
    m = find_min1()
    print(m)
decom(l, L)