import numpy as np
from collections import Counter
import itertools

# 计算成本
def calc_cost(loss, joint, l_size):
    '''
    loss: 废料长度
    joint: 接头数量
    l_size: 钢筋直径
    '''
    return l_size**2*loss/1000*0.00617*2000 + joint*10

# 由剩余未匹配的钢筋数量计算成本
# 这里直接最大化成本，方便后续的拟合
def calc_cost_by_unmatched(need_num, l, need_len, l_size=32):
    '''
    need_num: 匹配完毕后剩余未匹配的钢筋数量 = need - sum(solution)
    l: 原始钢筋定长
    need_len: 目标钢筋长度
    l_size: 钢筋的直径
    '''
    loss = 0
    for i in range(len(need_num)):
        if need_num[i]>0:   # 剩余未匹配的钢筋数量
            loss += float((l-need_len[i])*need_num[i])
        else:               # 多匹配的钢筋数量
            loss += float(l*-need_num[i])
    return calc_cost(loss, 0, l_size)    

# 计算当前组合最终完成的长度
# solution: [0,...,] 表示选择该种组合的选择数量,长度为patterns的长度
# patterns: 所有组合的列表，每个元素为 [counter, loss, joint, cost, eer, combin]
def calc_completion_lenghts(solution, need, patterns):
    hascut_lengths = np.zeros_like(need)
    for i in patterns:
        hascut_lengths += patterns[i][0]*solution[i]
    return hascut_lengths

# 计算当前组合的余料和接头数量
def calc_loss_joint(combination, l, l_min):
    '''
    combination: 组合列表，[3000,3000,4000,...]
    l: 原始钢筋定长
    l_min: 最小接头数量
    '''
    loss = 0
    joint = 0
    _l = l
    for length in combination: 
        if _l<length: 
            if _l<l_min:    # 剩余长度小于最小接头数量
                loss += _l
                _l = l
            else:           # 用接头连接下一个钢筋
                _l += l 
                joint += 1
        _l -= length
    # 计算余料                        
    if _l<l: loss += _l
    return loss, joint

# 产生patterns，最低1个组合，最高max_len个组合
# 计算废料和接头数量，计算成本，计算能效比
# patterns: 所有组合的辞典，key为索引 每个元素为 [counter, loss, joint, cost, eer, combin]
# counter : 组合计数 [1,2,0]
# loss ：废料长度
# joint： 接头数量
# cost： 废料+接头的成本
# err: 能效比 cost/sum(counter)
# list: 组合列表["L1","L1","L3",....]
def pattern_oringin_by_sampling(l, L, sampling_count, max_len=10, l_min=200, l_size=32, include_less=True):
    '''
    l: 原始钢筋定长
    L: 目标钢筋长度
    sampling_count: 采样的数量
    max_len: 最大组合数
    l_min: 最小接头数量
    l_size: 接头大小
    include_less: 是否包含长度小于L的组合
    '''
    patterns = {}
    # 完整组合
    patterns_list = []
    # 保留组合
    patterns_list_keep = []
    for i in range(1, max_len):
        # 按组合数产生组合
        combinations = itertools.product(L, repeat=i)
        for combination in combinations:   
            combination_values=[L[key] for key in combination]

            # 计算接头数量和余料
            loss, joint = calc_loss_joint(combination_values, l, l_min)
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
            # 如果有0的count，则保留在patterns_list_loss中，否则保留在patterns_list中  
            if include_less and np.any(counter==0) and np.all(counter<=1):
                patterns_list_keep.append([counter, loss, joint, cost, eer, combination])
            else:
                patterns_list.append([counter, loss, joint, cost, eer, combination])

    print("完整组合数：", len(patterns_list))
    print("保留组合数：", len(patterns_list_keep))
    # 按成本排序
    patterns_list = sorted(patterns_list, key=lambda x:x[3])
    # 取前sampling_count-len(patterns_list_keep)个
    patterns_list = patterns_list[:sampling_count-len(patterns_list_keep)]
    # 合并完整和不完整的pattern
    patterns_list += patterns_list_keep
    # 转换成dict
    for i, pattern in enumerate(patterns_list):
        patterns[i] = pattern

    return patterns   


# 产生patterns，最低1个组合，最高max_len个组合
# 计算废料和接头数量，计算成本，计算能效比
# patterns: 所有组合的辞典，key为索引 每个元素为 [counter, loss, joint, cost, eer, combin]
# counter : 组合计数 [1,2,0]
# loss ：废料长度
# joint： 接头数量
# cost： 废料+接头的成本
# err: 能效比 cost/sum(counter)
# list: 组合列表["L1","L1","L3",....]
def pattern_oringin_by_loss(l, L, max_loss, max_len=10, l_min=200, l_size=32):
    '''
    l: 原始钢筋定长
    L: 目标钢筋长度
    max_loss: 最大废料损失
    max_len: 最大组合数
    l_min: 最小接头数量
    l_size: 接头大小
    '''
    # 完整组合
    patterns_list = []
    # 保留组合
    patterns_list_keep = []
    for i in range(1, max_len):
        # 按组合数产生组合
        combinations = itertools.product(L, repeat=i)
        for combination in combinations:               
            combination_values=[L[key] for key in combination]
            # 计算接头数量和余料
            loss, joint = calc_loss_joint(combination_values, l, l_min)

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
                
                if np.any(counter==0):
                    patterns_list_keep.append([counter, loss, joint, cost, eer, combination])
                else:
                    patterns_list.append([counter, loss, joint, cost, eer, combination])


    # 记录pattern和loss，返回dict {idx: [pattern, loss]}                                                        
    patterns = {}
    patterns_tail = {}
    for i, pattern in enumerate(patterns_list):
        patterns[i] = pattern

    for i, pattern in enumerate(patterns_list_keep):
        patterns_tail[i] = pattern

    print("完整组合数：", len(patterns))
    print("尾料组合数：", len(patterns_tail))

    return patterns, patterns_tail   
