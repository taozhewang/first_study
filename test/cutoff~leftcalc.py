import numpy as np

def calc_left(left, l, L, joint, waste_cost, paste_cost): # 用于计算剩余需要填补的目标材料产生的成本
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
    cost = waste * waste_cost + paste * paste_cost
    return cost, waste, paste
