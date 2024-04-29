import numpy as np
from collections import Counter
import itertools

# 计算成本
def calc_cost(loss, joint, l_size):
    return l_size**2*loss/1000*0.00617*2000 + joint*10

# 按照普通拼接计算长度产生的废料和接头数量
def calc_loss_joint(need_num, l, need_len, l_min=200):
    '''
    need_num: 目标钢筋数量
    l: 原始钢筋定长
    need_len: 目标钢筋长度
    l_min: 最小废料长度
    '''
    loss = 0
    joint = 0
    _l = l
    for i in range(need_num):
        if need_num[i]>0:
            # 依次计算每个钢筋的接头数量和余料
            for j in range(need_num[i]):
                if _l >= need_len[i]:
                    _l -= need_len[i]
                else:
                    if _l < l_min:
                        loss += _l
                        _l = l
                    else:
                        _l += l 
                        joint += 1
    if _l<l: loss += _l
    return loss, joint

# 产生patterns，最低1个组合，最高max_len个组合
# 计算余料和接头数量，计算成本，计算能效比
# patterns: 所有组合的列表，每个元素为 [{"L1":count,"L2":count,...}, loss, joint, cost, eer]
def pattern_oringin(l, L, max_loss, max_len=10, l_min=200, l_size=32):
    '''
    l: 原始钢筋定长
    L: 目标钢筋长度
    max_loss: 最大废料损失
    max_len: 最大组合数
    l_min: 最小接头数量
    l_size: 接头大小
    '''
    patterns = {}
    idx = 0
    for i in range(1, max_len):
        # 按组合数产生组合
        combinations = itertools.product(L, repeat=i)
        for combination in combinations:   
            # 计算接头数量和余料
            loss = 0
            # 计算接头数量
            joint = 0
            _l = l
            for key in combination: # ["L1","L2","L3"]
                if _l>=L[key]:
                    _l -= L[key]
                else:
                    if _l<l_min:
                        loss += _l
                        _l = l
                    else:
                        _l += l 
                        joint += 1
            # 计算余料                        
            if _l<l: loss += _l

            # 这里过滤余料大于losses1的组合，需要保留为1，2的组合，方便尾料的处理
            if loss <= max_loss or i < len(L):
                # 统计组合中的数量，返回dict {key1:count,key2:count,...}
                # 计算成本
                cost = calc_cost(loss, joint, l_size)
                # 计算能效比
                eer = cost/len(combination)

                # 这里的key需要和L的key对应，如果不在组合中，则count为0
                combination_counter = Counter(combination)                    
                pattern={}
                for key in L:
                    if key in combination_counter:
                        pattern[key] = combination_counter[key]
                    else:
                        pattern[key] = 0
                
                # 记录pattern和loss，返回dict {idx: [pattern, loss]}                                                        
                patterns[idx]=[pattern, loss, joint, cost, eer]
                idx += 1
    return patterns   