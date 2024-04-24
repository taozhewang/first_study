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
# def accum(L, need, result):
#     k = [[[], 12000]]
#     for i in result.values():
#         k.append(i)
#         if k[len(k) - 2][1] < k[len(k) - 1][1]:
#             del k[len(k) - 1]
#         elif k[len(k) - 2][1] > k[len(k) - 1][1]:
#             del k[len(k) - 2]
#     print(k)
#     count = np.array([0 for _ in L.keys()])
#     while True:
#         lk = len(k)
#         turn = np.random.choice(lk)
#         # print(k[turn])
#         turn = k[turn]
#         count = count + np.array(turn[0])
#         # print(count)
#         if not all(count[i] < need[i] for i in range(len(need))):
#             return count - np.array(turn[0])
# p = np.array(accum(L, need, result))
# print(p)
# need2 = need - p
# print(need2)
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
    # usage = {}
    # for i in range(len(op)):
    #     usage[i] = 0
    def ratio(op, need):
        ratio_need = np.array(need / np.sum(need))
        possibility_list = np.array([])
        for i in op:
            ratio_type = np.array(i / np.sum(i))
            ratio_distance = np.sum((ratio_type - ratio_need) ** 2)
            # print(ratio_distance)
            possibility_list = np.append(possibility_list, ratio_distance)
        po1 = 1 / possibility_list ** 4
        # print(po1)
        po2 = po1 / np.sum(po1)
        # print(po2)
        return po2
    curr_need = copy.deepcopy(need)
    while True:
        r = ratio(op, curr_need)
        # print(op, r)
        choose = np.random.choice(range(len(op)), p = r)
        curr_need = curr_need - op[choose]
        if any(curr_need < 0):
            return curr_need + op[choose], fake_op
        fake_op[convert[choose]] += 1
left, acc = accum2()

print(left, acc)
# ac = np.array([0, 0, 0])
# for i in acc.keys():
#     ac += np.array(result[i][0]) * acc[i]
#     print(ac)
accumulator = {}
for i in result.keys():
    accumulator[i] = 0

ways = {}
def accum3(result, left, accumulator, pointer, countor):
    null_space = np.where(left > 0)
    # print(null_space)
    null_number = np.array([result[pointer][0][i] for i in null_space[0]])
    # print(null_number)
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
# print(ways)
def find_min1():
    # print(result)
    more = []
    for way in ways.values():
        # print(way, acc)
        total_sum = {}
        for i in ways.keys():
            # print(way, acc)
            total_sum[i] = way[i] + acc[i]
        # print(total_sum)
        materials = np.array([0 for _ in range(len(need))])
        waste = 0
        for i in total_sum.keys():
            
            materials += np.array(result[i][0]) * total_sum[i]
            waste += result[i][1] * total_sum[i]
            # print(materials)
        over_produce = materials - need
        # print(over_produce, need)
        L_length = list(L.values())
        over_p = 0
        for i in range(len(over_produce)):
            over_p += L_length[i] * over_produce[i]
        more.append(np.array([over_p, waste]))
    return more
m = find_min1()
print(m)

#%%
def decom(l, L):
# 第一步：从L中允许重复取样，只要满足L的总size小于l的组合   
    result1 = set()
    def count_decomposition1(dst_dict, temp_sum, temp_combo, max_sum, result):
    # 如果当前和大于 max_sum,则剪枝
        if temp_sum > max_sum:
            return
        # 如果当前和大于0,则将当前组合添加到结果中,并去重
        if temp_sum>0:
            result.add(tuple(sorted(temp_combo)))
        # 回溯
        for key in L:
            temp_combo.append(key)
            count_decomposition1(dst_dict, temp_sum + L[key], temp_combo, max_sum, result)
            temp_combo.pop()
    count_decomposition1(L, 0, [], l, result1)
    print(result1)

    result2 = set()
    def count_decomposition2(dst_dict, temp_sum, temp_combo, max_sum, result):
    # 如果当前和大于 max_sum,则加入并剪枝
        if temp_sum > max_sum:
            result.add(tuple(sorted(temp_combo)))
            return
        # 回溯
        for key in L:
            temp_combo.append(key)
            count_decomposition2(dst_dict, temp_sum + L[key], temp_combo, max_sum, result)
            temp_combo.pop()
    count_decomposition2(L, 0, [], l, result2)
    print(result2)


    # 第三步：从L中允许重复取样，找出所有满足L的总size小于2*l的组合
    result3_medium = set()
    count_decomposition1(L, 0, [], 2*l, result3_medium)
    print(result3_medium)

    # 防止出现不能由尾料减尾尾料得到的组合
    result3 = set()
    for r in result3_medium:
        if r not in result1 and r not in result2:
            result3.add(r)
    print(result3)
    return result1, result2, result3
decom(l, L)