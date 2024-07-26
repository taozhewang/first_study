import numpy as np
import copy
import time

def solution_initialize(patterns: list, taboo_size: int):# 产生一个初始解
    solutions = np.zeros((taboo_size, len(patterns)))
    return solutions # 矩阵

def solution_update(solutions: np.ndarray, taboo_size: int, need: np.ndarray, patterns: list, variation_count: int, patterns_p: list): # 更新解，即寻找解的邻域
    for row in range(taboo_size):
        ids = np.random.choice(np.size(solutions, 1), variation_count, replace = False, p = patterns_p) # 根据成本大小选择需要变动的组合，选择个数为variation_count
        for idx in ids:
            k = np.random.random(1) # 随机加一减一处理
            if k >= 0.5:
                solutions[row][idx] += 1
            else:
                solutions[row][idx] -= 1
        solutions[row][solutions[row] < 0] = 0 # 避免出现个数为负数的情况

        accumulate = np.zeros(len(need))
        for idx, num in enumerate(solutions[row]):
            accumulate += patterns[idx] * num
            
        while any(accumulate > need): # 将当前解内所有目标个数维持到需求以下
            c = np.where(solutions[row] > 0)[0]
            # if len(c) == 0: break
            k = np.random.choice(c)
            if solutions[row][k] > 0: 
                solutions[row][k] -= 1
                accumulate -= patterns[k]
    return solutions

def calc_left(left: np.ndarray, l: float, L: np.ndarray, joint: float, waste_cost: float, paste_cost: float): # 用于计算剩余需要填补的目标材料产生的成本
    length = 0
    waste = 0
    paste = 0
    for idx, num in enumerate(left):                    # 按照顺序排序计算成本，并非最小成本
        if num == 0: continue
        for _ in range(int(num)):
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
    cost = waste * waste_cost + paste * paste_cost # 计算成本
    return cost, waste, paste

def solution_quality(solutions: np.ndarray, need: np.ndarray, patterns: list, patterns_property: list, 
                     l: float, L: np.ndarray, joint: float, waste_cost: float, paste_cost: float): # 计算当前解的成本
    C = np.zeros(np.size(solutions, 0))
    W = np.zeros(np.size(solutions, 0))
    P = np.zeros(np.size(solutions, 0))
    for s in range(np.size(solutions, 0)):
        accumulate = np.zeros(len(need))
        for p, num in enumerate(solutions[s]):
            C[s] += num * patterns_property[p][0]
            W[s] += num * patterns_property[p][1]
            P[s] += num * patterns_property[p][2]
            accumulate += patterns[p] * num
        
        left_cost, left_waste, left_paste = calc_left(need - accumulate, l, joint, waste_cost, paste_cost)
        C[s] += left_cost
        W[s] += left_waste
        P[s] += left_paste

    return C, W, P

