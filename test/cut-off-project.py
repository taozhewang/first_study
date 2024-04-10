#%%
# cutoff
import numpy as np
# L 为目标，type 为 dict
# l 为原料
# attention: L 必须升序排列，不然第三部分会出错（代码是默认升序写的）
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
l = 12000

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
decom(l, L)
# 返回：所有可能出现的组合
#%%