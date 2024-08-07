import numpy as np
from collections import Counter, deque
import itertools
import random
import pickle,os
import copy

# 计算成本
def calc_cost(loss, joint, l_size):
    '''
    loss: 废料长度
    joint: 接头数量
    l_size: 钢筋直径
    '''
    return round(l_size**2*loss/1000*0.00617*2000 + joint*10,3)

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
        while _l<length: 
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
def pattern_oringin4(l, L, max_len=10, l_min=200, l_size=32):
    '''
    l: 原始钢筋定长
    L: 目标钢筋长度
    max_len: 最大组合数
    l_min: 最小接头数量
    l_size: 接头大小
    include_less: 是否包含长度小于L的组合
    '''
    
    # 查看cache文件是否存在，如果有直接返回
    k = "_".join([str(L[key]) for key in L])
    fname = f"{l}_{k}_{max_len}_{l_min}_{l_size}.pkl"    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(script_dir, fname)
    if os.path.exists(fname):
        with open(fname,'rb') as f:
            return pickle.load(f)
        
    patterns = {}
    # 完整组合
    patterns_list = []
    # 如果有重复类别，只保留余料最小的的组合 {count:cost}
    patterns_saved = {}
    for i in range(1, max_len+1):
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

            counter_str = "_".join([str(v) for v in counter])      
            # 清除高余料的组合
            if counter_str in patterns_saved:
                if loss<l_min and cost!=patterns_saved[counter_str][0] and patterns_saved[counter_str][-2]<l_min:
                    if joint>patterns_saved[counter_str][-1]:
                        combination_str = '_'.join([str(v) for v in patterns_saved[counter_str][-3]])
                    else:
                        combination_str = '_'.join([str(v) for v in combination])
                    find = False
                    for x in patterns_saved:
                        combination1_str = '_'.join([str(v) for v in patterns_saved[x][-3]])
                        if  combination_str.find(combination1_str)>=0 and len(combination1_str)<len(combination_str):
                            print(combination_str, combination1_str, patterns_saved[x][-2], patterns_saved[x][-1])
                            find=True
                            break   
                    if not find:
                        print(patterns_saved["1_4_0"])
                        print(counter,"L:",L,"l:",l,"loss:",l_min)
                        print(combination, loss, joint )
                        print(patterns_saved[counter_str][-3],patterns_saved[counter_str][-2],patterns_saved[counter_str][-1])
                        input()
                if cost>=patterns_saved[counter_str][0]: continue
                # 如果新的比旧的成本低，删除旧的
                del patterns_list[patterns_saved[counter_str][1]]
            # 记录下当前成本和位置
            patterns_saved[counter_str]=(cost, len(patterns_list), combination, loss, joint)
            patterns_list.append([counter, loss, joint, cost, eer, combination])

    patterns_list_len = len(patterns_list)
    print("组合数：", patterns_list_len)
        
    patterns_list = sorted(patterns_list, key=lambda x:x[3])

    # 转换成dict
    for i, pattern in enumerate(patterns_list):
        patterns[i] = pattern

    # 保存数据到cache文件
    # with open(fname,'wb') as f:
    #     pickle.dump(patterns, f)

    return patterns   

# 产生组合：l 原料长度 L 目标类型 radius 最大耗用原料数 l_min 最大可以拼接长度 l_size 原料直径
# patterns: 所有组合的辞典，key为索引 每个元素为 [counter, loss, joint, cost, stage, combin]
def pattern_oringin3(l, L, radius, l_min=200, l_size=32):
    L_keys = list(L.keys())
    L_values = list(L.values())
    L_length = len(L)
    patterns = {}
    patterns_list=[]                            # 存放切割方案的切割情况
    accumulator = [0 for _ in L_values]         # 存放当前切割方案的切割情况
    filter_duplicates = {}
    # accumulator 计数, patterns_path 组合顺序,  pointer 当前类型, length 余料, paste 接头, stage 耗用
    def _recurse(perm, accumulator, patterns_path, pointer, length, paste, stage):
        # 达到最多切割原料根数，结束递归返回
        # if perm[0]==1:
        #     print(perm, patterns_path, accumulator, pointer, length, paste, stage)
        #     input()
        if stage > radius: return                
        if length < l_min:      # 达到最小组合方案，返回
            # 检查重复取最小成本
            cost = calc_cost(length, paste, l_size)
            counter_str = "_".join([str(v) for v in accumulator])   
            if counter_str in filter_duplicates:
                if cost>=filter_duplicates[counter_str][0]:                           
                    return
                del patterns_list[filter_duplicates[counter_str][1]]
            filter_duplicates[counter_str]=(cost, len(patterns_list))
            patterns_list.append([np.array(accumulator,dtype=int), length, paste, cost, stage, patterns_path[:]]) 
            return
            
        for i in range(pointer, L_length):
            p = perm[i]
            prev_state = [length, paste, stage]            
            while length < L_values[p]:
                length += l             # 取一根原料拼接：长度+l
                paste += 1              # 接头+1
                stage += 1              # 使用原料数+1
            length -= L_values[p]       # 减去原料的长度，得到尾料的长度
            accumulator[p] += 1         # 记录增加当前目标的组合次数
            patterns_path.append(L_keys[p])     # 记录当前方案
            _recurse(perm, accumulator, patterns_path, i, length, paste, stage )    # 部分递归,设置参数i为0为全递归，但太慢了，没法用
            patterns_path.pop()         # 如果递归结束，回退一步，找其他组合    
            accumulator[p] -= 1 
            length, paste, stage = prev_state 

    for perm in itertools.permutations(range(L_length)): 
        _recurse(perm, accumulator, [], 0, l, 0, 1)   # 调用递归找组合，初始化从一根原料开始 

    patterns_list = sorted(patterns_list, key=lambda x:x[3])      # 按成本排序
    for i, pattern in enumerate(patterns_list): # 转换成dict
        patterns[i] = pattern
    return patterns


def pattern_oringin2(l, L, radius, l_min=200, l_size=32):
    L_length = np.array(list(L.values()))
    patterns = {}
    patterns_list=[]                            # 存放切割方案的切割情况
    accumulator = np.zeros(len(L),dtype=int)    # 存放当前切割方案的切割情况
    def _recurse(accumulator, patterns_path, length, cut, paste, pointer, stage):

        # 达到最多切割原料根数，结束递归返回
        if stage == radius:
            return
        
        Re_accumulator = copy.deepcopy(accumulator)             # 递归时需要复制一份accumulator
        Re_patterns_path = copy.deepcopy(patterns_path)             # 递归时需要复制一份patterns_path

        if pointer < len(L_length) - 1:         # 还可以递归，继续开新的切割递归
            _recurse(Re_accumulator, Re_patterns_path, length, cut, paste, pointer + 1, stage)
            
        length += L_length[pointer]         # 尾料长度+当前选择的目标的长度
        Re_accumulator[pointer] += 1        # 记录增加当前目标的组合次数
        Re_patterns_path.append(pointer)    # 记录当前方案
        if length > l:                      # 当前现有大于原料长度，需要补1根原料进行切割
            length = length - l             # 减去原料的长度，得到尾料的长度
            stage += 1                      # 使用原料次数+1
            cut += 1                        # 增加切割次数
            paste += 1                      # 增加拼接次数  
        elif length == l:                   # 刚好达到原料长度，记录并继续递归
            length = 0                      # 切割完后，尾料长度清零
            cost = calc_cost(cut, paste, l_size)
            patterns_list.append([Re_accumulator, 0, paste, cost, stage + 1, Re_patterns_path])                # 记录切割方案
            stage += 1                       # 使用原料次数+1
            pointer = 0                      # 切割完后，指针回到开头，继续递归
        else:                               # 剩余长度小于原料长度，继续递归
            cut += 1                        # 增加切割次数
            left = l - length               # 剩余长度
            if left <= l_min:             # 剩余长度小于等于最低损耗，不用补原料
                cost = calc_cost(left, paste, l_size)
                patterns_list.append([Re_accumulator, left, paste, cost, stage + 1, Re_patterns_path])                # 记录切割方案

        _recurse(Re_accumulator, Re_patterns_path, length, cut, paste, pointer, stage)    # 递归
       
    _recurse(accumulator, [], 0, 0, 0, 0, 0)

    patterns_list_len = len(patterns_list)
    print("组合数：", patterns_list_len)
        
    patterns_list = sorted(patterns_list, key=lambda x:x[3])

    # 转换成dict
    for i, pattern in enumerate(patterns_list):
        patterns[i] = pattern

    return patterns

# 按长度足一由小到大便利，但效率太低
def pattern_oringin(l, L, radius, l_min=200, l_size=32):
    L_keys = list(L.keys())
    L_values = list(L.values())
    L_length = len(L)
    patterns = {}
    patterns_list=[]                            # 存放切割方案的切割情况
    accumulator = [0 for _ in L_values]         # 存放当前切割方案的切割情况
    filter_duplicates = {}
    
    stack = list()
    for i in range(L_length):
        accumulator = [0 for _ in L_values]
        stack.append([accumulator, [], i, l, 0, 1])
    count=0
    while stack:
        accumulator, patterns_path, next_pointer, length, paste, stage = stack.pop()
        count+=1
        if count%1000000==0:
            print(count, len(stack), len(patterns_list), accumulator, patterns_path, next_pointer, length, paste, stage)
        if stage > radius: continue                
        if length < l_min:      # 达到最小组合方案，返回
            # 检查重复取最小成本
            cost = calc_cost(length, paste, l_size)
            counter_str = "_".join([str(v) for v in accumulator])   
            if counter_str in filter_duplicates:
                if cost>=filter_duplicates[counter_str][0]:                           
                    continue
                del patterns_list[filter_duplicates[counter_str][1]]
            filter_duplicates[counter_str]=(cost, len(patterns_list))
            patterns_list.append([np.array(accumulator,dtype=int), length, paste, cost, stage, patterns_path[:]]) 
            continue

        while length < L_values[next_pointer]:
            length += l             # 取一根原料拼接：长度+l
            paste += 1              # 接头+1
            stage += 1              # 使用原料数+1
        length -= L_values[next_pointer]       # 减去原料的长度，得到尾料的长度
        accumulator[next_pointer] += 1         # 记录增加当前目标的组合次数
        patterns_path.append(L_keys[next_pointer])     # 记录当前方案
        for i in range(L_length):
            stack.append([accumulator[:], patterns_path[:], i, length, paste, stage])    # 完全递归            
  
    patterns_list = sorted(patterns_list, key=lambda x:x[3])      # 按成本排序
    for i, pattern in enumerate(patterns_list): # 转换成dict
        patterns[i] = pattern
    return patterns


if __name__ == "__main__":    
    # for l in range(13,50):
    #     print(l)
    #     pattern_oringin(l, {'L1' : 9, 'L2' : 8, 'L3' : 12,}, max_len=10, l_min=2, l_size=32)

    l = 12000       # 原始钢筋长度
    l_size = 32     # 钢筋的规格
    l_limit_len = 200   # 钢筋的最小可利用长度
    radius = 6   # 最大组合数
    L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}     # 目标钢筋长度
    need = np.array([552, 658, 462])    # 目标钢筋的数量
    patterns = pattern_oringin3(l, L, radius, l_limit_len, l_size)
    
    for i in patterns:
        print(f"{i}: {patterns[i][0]} 余料: {patterns[i][1]} 接头: {patterns[i][2]} 成本: {patterns[i][3]} 原料: {patterns[i][4]} 路径: {patterns[i][5]}")

    # combination = []
    # for i,key in enumerate(L):
    #     combination += [L[key]] * need[i]
    
    # # 随机打散组合
    # random.shuffle(combination)
    # loss, joint = calc_loss_joint(combination, l, l_limit_len)

    # reward = calc_cost(loss, joint, l_size)
    # print(f"钢筋总根数：", len(combination))        
    # print(f"成本: {reward} \t接头: {joint} \t剩余: {loss}")