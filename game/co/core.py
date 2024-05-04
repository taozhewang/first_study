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
    for i in range(len(need_num)):
        if need_num[i]>0:
            # 依次计算每个钢筋的接头数量和余料
            for j in range(need_num[i]):
                if _l < need_len[i]:
                    if _l < l_min:
                        loss += _l
                        _l = l
                    else:
                        _l += l 
                        joint += 1
                _l -= need_len[i]
    if _l<l: loss += _l
    return loss, joint

# 计算当前组合最终完成的长度
# solution: [0,...,] 表示选择该种组合的选择数量,长度为patterns的长度
# patterns: 所有组合的列表，每个元素为 [counter, loss, joint, cost, eer, combin]
def calc_completion_lenghts(solution, need, patterns):
    hascut_lengths = np.zeros_like(need)
    for i in patterns:
        hascut_lengths += patterns[i][0]*solution[i]
    return hascut_lengths

# 产生patterns，最低1个组合，最高max_len个组合
# 计算废料和接头数量，计算成本，计算能效比
# patterns: 所有组合的辞典，key为索引 每个元素为 [counter, loss, joint, cost, eer, combin]
# counter : 组合计数 [1,2,0]
# loss ：废料长度
# joint： 接头数量
# cost： 废料+接头的成本
# err: 能效比 cost/sum(counter)
# list: 组合列表["L1","L1","L3",....]
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
            combination=list(combination)
            # 计算接头数量和余料
            loss = 0
            # 计算接头数量
            joint = 0
            _l = l
            for key in combination: # ["L1","L2","L3"]
                if _l<L[key]:
                    if _l<l_min:
                        loss += _l
                        _l = l
                    else:
                        _l += l 
                        joint += 1
                _l -= L[key]
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
                counter=np.zeros(len(L), dtype=int)
                for i, key in enumerate(L):
                    if key in combination_counter:
                        counter[i] = combination_counter[key]
                    else:
                        counter[i] = 0
                
                # 记录pattern和loss，返回dict {idx: [pattern, loss]}                                                        
                patterns[idx]=[counter, loss, joint, cost, eer, combination]
                idx += 1
    return patterns   

