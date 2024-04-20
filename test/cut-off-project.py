#%%
# cutoff
import numpy as np
# L 为目标，type 为 dict
# l 为原料
# attention: L 必须升序排列，不然第三部分会出错（代码是默认升序写的）
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
l = 12000
need = np.array([[852, 660, 162]])

def decom(l, L):
    # 第一步
    def count_decomposition1(l, L, count, pointer):
        L_keys = list(L.keys())
        L_values = list(L.values())
        if len(L_keys) == pointer:
            return [count]
        elif l < L_values[pointer]:
            return count_decomposition1(l, L, count, pointer + 1)
        else:
            return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                        pointer) + count_decomposition1(l, L, count, pointer + 1)
    result1 = count_decomposition1(l, L, [], 0)
    # 防止出现[]的情况
    del result1[len(result1) - 1]
    print(result1)

    # 第二步
    def count_decomposition2(l, L, count, pointer):
        L_keys = list(L.keys())
        L_values = list(L.values())
        if l < 0:
            return [count]
        elif pointer == len(L_keys) - 1:
            return count_decomposition2(l - L_values[pointer], L, count + [L_keys[pointer]], pointer)
        else:
            return count_decomposition2(l - L_values[pointer], L, count + [L_keys[pointer]],
                                        pointer) + count_decomposition2(l, L, count, pointer + 1)
    result2 = count_decomposition2(l, L, [], 0)
    print(result2)

    # 第三步
    result3_medium = count_decomposition1(2 * l, L, [], 0)
    # 防止出现不能由尾料减尾尾料得到的组合
    l1 = min(len(l) for l in result1)
    l2 = min(len(l) for l in result2)
    l3 = l1 + l2
    result3 = []
    for i in result3_medium:
        if len(i) >= l3 and i not in result2:
            result3.append(i)
    print(result3)
    return result1, result2, result3
# decom(l, L)

# 返回：所有可能出现的组合
#%%
l = 12000
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}

def decom_quant(l, L):
    ways = {}
    def decom(l, L):
        def count_decomposition1(l, L, count, pointer):
            L_keys = list(L.keys())
            L_values = list(L.values())
            if len(L_keys) == pointer:
                return [count]
            elif l < L_values[pointer]:
                return count_decomposition1(l, L, count, pointer + 1)
            else:
                return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                            pointer) + count_decomposition1(l, L, count, pointer + 1)
        result1 = count_decomposition1(l, L, [], 0)
        del result1[len(result1) - 1]
        # print(result1)
        return result1
    def quant(l, L, a):
        for j in a:
            l = l - L[j]
        return l
    result = decom(l, L)
    count = 0
    for i in result:
        count += 1
        ways[count] = [i, quant(l, L, i)]
    return ways

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
p = 11
def decom_quant(l, L):
    ways = {}
    def decom(l, L):
        def count_decomposition1(l, L, count, pointer):
            L_keys = list(L.keys())
            L_values = list(L.values())
            if len(L_keys) == pointer:
                return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                            pointer)
            elif l < L_values[pointer]:
                return [count]
            else:
                return count_decomposition1(l - L_values[pointer], L, count + [L_keys[pointer]],
                                            pointer) + count_decomposition1(l, L, count, pointer + 1)
        result1 = count_decomposition1(l, L, [], 0)
        print(result1)
        return result1
 
    def quant(l, L, a):
        for j in a:
            l = l - L[j]
        return l
    result = decom(l, L)
    count = 0
    for i in result:
        count += 1
        ways[count] = [i, quant(l, L, i)]
    return ways

#%%
def decom_pattern(l, L):
    def count_decomposition1(l, L, count, pointer):
        L_keys = list(L.keys())
        L_values = list(L.values())
        if l < 100:
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
for i in range(1, 7):
    k = decom_pattern(i * l, L)
    for r in k.values():
        result[count] = r
        count += 1
print(result)
# R = np.array([i[0] for i in result.values()])
# R = R.T
# print(R)
# R_pseudo_inverse = np.linalg.pinv(R)
# ww = R_pseudo_inverse @ need.T
# print(ww)
K = {}
for i in result.keys():
    K[i] = np.array(result[i][0])
'''
K = {0: [1, 3, 4], 1: [5, 2, 4], 2: [1, 9, 1], 
     3: [8, 3, 3], 4: [5, 8, 1], 5: [12, 2, 3], 
     6: [8, 9, 0], 7: [3, 4, 9]}
'''
# to the sake of your computer, please do not let numbers in 'need' bigger than 10
need = np.array([4, 2, 3])
def count_decomposition2(need, K, count, pointer):
    K_keys = list(K.keys())
    K_values = list(K.values())
    low = np.where(need > 0)
    # print(pointer)
    minus = np.where(K_values[pointer] == 0)
    # print(need)
    if np.array_equal(low, minus) and len(K_keys) > pointer + 1:
        return count_decomposition2(need, K, count, pointer + 1)
    elif all(need <= 0):
        return [count]
    elif len(K_keys) == pointer + 1:
        return count_decomposition2(need - K_values[pointer], K, count + [K_keys[pointer]],
                                    pointer)
    else:
        return count_decomposition2(need - K_values[pointer], K, count + [K_keys[pointer]],
                                    pointer) + count_decomposition2(need, K, count, pointer + 1)
rresult = count_decomposition2(need, K, [], 0)
print(rresult)
#%%