import numpy as np
from collections import Counter
import itertools
import random
import pickle,os

# 计算成本
def calc_cost(loss, joint, l_size):
    '''
    loss: 废料长度
    joint: 接头数量
    l_size: 钢筋直径
    '''
    return l_size**2*loss/1000*0.00617*2000 + joint*10

# 由剩余未匹配的钢筋数量计算成本
# 这里直接最大化成本
def calc_cost_by_unmatched2(need_num, l, need_len, l_size=32, l_min=200):
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
    return calc_cost(loss, 0, l_size), loss, 0    

# 这里直接按顺序计算成本
def calc_cost_by_unmatched(need_num, l, need_len, l_size=32, l_min=200):
    '''
    need_num: 匹配完毕后剩余未匹配的钢筋数量 = need - sum(solution)
    l: 原始钢筋定长
    need_len: 目标钢筋长度
    l_size: 钢筋的直径
    '''
    _loss = 0
    combination = []
    for i in range(len(need_num)):
        if need_num[i]>0:   # 剩余未匹配的钢筋数量
            combination += [need_len[i]]*need_num[i]
        else:               # 多匹配的钢筋数量
            _loss += -l*float(need_num[i])
    loss, joint = calc_loss_joint(combination, l, l_min)

    return calc_cost(loss+_loss, joint, l_size), loss, joint   

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

# 寻找当前组合的最小成本
def get_min_cost_combination(combination, l, l_min=200, l_size=32, max_iterations=1000000):
    '''
    combination: 组合列表，[3000,3000,4000,...]
    l: 原始钢筋定长
    l_min: 最小接头数量
    l_size: 钢筋的直径
    '''
    def backtrack(combination, path, result, visited):
        if len(path) == len(combination):
            loss, joint = calc_loss_joint(path, l, l_min)
            cost = calc_cost(loss, joint, l_size)
            result["calc_count"]+=1
            if result["calc_count"]%100000==0:
                print(result["calc_count"], cost)
            if cost < result["min_cost"]:
                result["min_cost"] = cost
                result["min_combination"]=path.copy()
                print("找到一个最优解:",cost)
            return

        if result["calc_count"]>=max_iterations: 
            return

        for i in range(len(combination)):
            if visited[i] or (i > 0 and combination[i] == combination[i-1] and not visited[i-1]):
                continue
            visited[i] = True
            path.append(combination[i])
            backtrack(combination, path, result, visited)
            path.pop()
            visited[i] = False

    def find_combination(combination, min_cost):
        combination.sort()  # 对列表进行排序
        result = {"min_cost":min_cost, "min_combination":None, "calc_count":0}
        visited = [False] * len(combination)
        backtrack(combination, [], result, visited)
        return result

    result = find_combination(combination, np.inf)
    return result["min_cost"], result["min_combination"] 

# 产生patterns，最低1个组合，最高max_len个组合
# 计算废料和接头数量，计算成本，计算能效比
# patterns: 所有组合的辞典，key为索引 每个元素为 [counter, loss, joint, cost, eer, combin]
# counter : 组合计数 [1,2,0]
# loss ：废料长度
# joint： 接头数量
# cost： 废料+接头的成本
# err: 能效比 cost/sum(counter)
# list: 组合列表["L1","L1","L3",....]
def pattern_oringin_by_sampling(l, L, sampling_count=-1, max_len=10, l_min=200, l_size=32):
    '''
    l: 原始钢筋定长
    L: 目标钢筋长度
    sampling_count: 采样的数量
    max_len: 最大组合数
    l_min: 最小接头数量
    l_size: 接头大小
    include_less: 是否包含长度小于L的组合
    '''
    
    # 查看cache文件是否存在，如果有直接返回
    k = "_".join([str(L[key]) for key in L])
    fname = f"{l}_{k}_{sampling_count}_{max_len}_{l_min}_{l_size}.pkl"    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(script_dir, fname)
    if os.path.exists(fname):
        with open(fname,'rb') as f:
            return pickle.load(f)
        
    patterns = {}
    # 完整组合
    patterns_list = []
    # 如果有重复类别，只保留一种 {cost:[count]}
    patterns_saved = {}
    for i in range(1, max_len):
        # 按组合数产生组合
        combinations = itertools.product(L, repeat=i)
        for combination in combinations:   
            combination = list(combination)
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
            
            # 清除相同的组合
            if cost not in patterns_saved:
                patterns_saved[cost]=[]
            findSame=False
            for c in patterns_saved[cost]:
                if np.array_equal(c, counter):
                    findSame=True
                    break
            if findSame: continue
            patterns_saved[cost].append(counter)
            
            # 记录pattern和loss，返回dict {idx: [pattern, loss]}
            # 如果有0的count，则保留在patterns_list_loss中，否则保留在patterns_list中  
            patterns_list.append([counter, loss, joint, cost, eer, combination])

    patterns_list_len = len(patterns_list)
    print("组合数：", patterns_list_len)
    
    # 采样数
    patterns_list = sorted(patterns_list, key=lambda x:x[3])
    if sampling_count>0:
        part_len = sampling_count-patterns_list_len
        if part_len>0:     # 如果当前记录不足，填充
            p = np.array([float(1/i[3]) for i in patterns_list])
            p = p/p.sum()
            patterns_list += random.choices(patterns_list, k=part_len, weights=p)
        elif part_len<0:   # 如果超过了，截断 
            # 按成本排序                                                                        
            patterns_list = patterns_list[:sampling_count]     

    # 转换成dict
    for i, pattern in enumerate(patterns_list):
        patterns[i] = pattern

    # 保存数据到cache文件
    with open(fname,'wb') as f:
        pickle.dump(patterns, f)

    return patterns   


if __name__ == "__main__":    
    l = 12000       # 原始钢筋长度
    l_size = 32     # 钢筋的规格
    l_limit_len = 200   # 钢筋的最小可利用长度
    L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}     # 目标钢筋长度
    need = np.array([552, 658, 462])    # 目标钢筋的数量
    combination = []
    for i,key in enumerate(L):
        combination += [L[key]] * need[i]
    
    # 随机打散组合
    random.shuffle(combination)
    
    loss, joint = calc_loss_joint(combination, l, l_limit_len)
    reward = calc_cost(loss, joint, l_size)
    print(f"钢筋总根数：", len(combination))        
    print(f"成本: {reward} \t接头: {joint} \t剩余: {loss}")