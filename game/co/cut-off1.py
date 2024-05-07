#%%
import numpy as np
from core import pattern_oringin_by_sampling, get_min_cost_combination, calc_loss_joint
import copy

'''
用相似度去扣减钢筋，使得总长度接近目标长度

废料长度: 11100.0
接头数量: 456.0
总成本: 144821.376
'''

# 原始钢筋长度
l = 12000
# 钢筋的规格
l_size = 32
# 钢筋最小尺寸
l_min = 200
# 目标钢筋长度
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# 目标钢筋的数量
need = np.array([552, 658, 462])
# 最大的组合数
radius = 14


'''
按最小余料，找出最长 radius 的组合，并计算其余料
返回值：
idx: 组合的id: [pattern: {} 组合的种类:数量, loss: 组合的余料, joint: 接头的数量, cost: 成本, eer: 能效比=cost/nums]
patterns:{idx: [pattern, loss, joint, cost, eer]} 
'''
def decom(l, L):
    
    # 求各种组合的列表
    print(f"create patterns (size: {radius})...")
    patterns = pattern_oringin_by_sampling(l, L, -1, radius)
    '''patterns: { 0: [[0,1,0], 0, 0, 0, 0,  ["L2"]],
                   1: [[1,0,1], 50,3,400,100,["L1","L3"]]} '''
    patterns_length = len(patterns)
    print(f"patterns[0]:", patterns[0])
    print(f"patterns[{patterns_length-1}]:", patterns[patterns_length-1])
    print(f"patterns length: {patterns_length}")

    # 求组合的的使用情况
    def accum2(patterns):
        # for calculating how many patterns are used
        op = []
        # fake_op用于统计在逼近need时所用的pattern的种类及个数
        fake_op = np.zeros(len(patterns))
        
        # op id 转换为 pattern id
        op_2_pattern = []
        # 计算 0.01% 分位数
        eer_percentile  = np.percentile([patterns[key][3] for key in patterns], 0.01)
        for key in patterns:
            counter, loss, joint, cost, eer, combin = patterns[key]   

            # 这里按照小于等于能效比来选择pattern：
            if eer <= eer_percentile:
                op.append(counter)
                op_2_pattern.append(key)
        '''
            fake_op: [0,1,0,2,......] 
            op: [array([5, 2, 4]), array([5, 8, 1])]
            convert: {0: 1, 1: 4, 2: 5, 3: 8}
        '''
        assert len(op) > 0, "no pattern can be used"
        print("op length:", len(op))
        
        # 方法：比例 + 随机， 目的是用尾料最少的pattern组装或逼近所需的目标数目
        def ratio(op, need):
            #  目标占比
            ratio_need = need / np.sum(need)
            
            # 所有组合和目标比例的欧式距离
            # possibility_list: [0.12809917 0.09141508 0.36264978]
            possibility_list = np.zeros(len(op), dtype=np.float32)
            for i, counter in enumerate(op):
                # 提高各组分比例离目标比例最近的pattern被选中的概率
                # pattern的比例
                ratio_type = counter / np.sum(counter)
                
                # pattern比例和目标need比例的欧式距离
                ratio_distance = np.sum((ratio_type - ratio_need) ** 2)
                
                if ratio_distance==0:
                    possibility_list[i] = 1e-6
                else:     
                    possibility_list[i] = ratio_distance
                
            # 相似度为距离的反比，取4次方只是因为逼近效果好一点
            po1 = 1 / possibility_list ** 4
            # 将相似度归一化为概率
            po2 = po1 / np.sum(po1)
            '''po2: [1.31134366e-01 1.09466700e-01 1.11207025e-03 1.31846626e-02
                    6.22324483e-03 6.02589438e-01 1.45555565e-05 1.50820746e-04
                    1.31134366e-01 4.11999268e-03 8.69783355e-04]'''
            return po2
        
        curr_need = copy.deepcopy(need) # 需要的目标数目
        op_idxs = list(range(len(op)))  # 所有op的id   
        while True:
            # 按 组合比例 和 剩余目标比例 的相似度 求 采样概率
            r = ratio(op, curr_need)
            
            # 按概率选择一个pattern进行叠加
            choose = np.random.choice(op_idxs, p = r)
            
            # 对当前的need进行削减
            temp_curr_need = curr_need - op[choose]
            
            # 截止条件：当前的剩余need出现有小于0的项时，返回 当前剩余need、所用的pattern的种类及个数
            if any(temp_curr_need < 0):                
                return curr_need, fake_op
            
            # 如果没有到截止条件，则计入选中的pattern
            curr_need = temp_curr_need
            fake_op[op_2_pattern[choose]] += 1
     
    # 剩余匹配量，当前patterns的使用量            
    left, acc = accum2(patterns)
    
    '''left: [2 0 1], 
    acc: [0,1,3,0,....]'''

    print("第一次匹配：")
    loss  = np.sum([num*patterns[i][1] for i,num in enumerate(acc)])
    joint = np.sum([num*patterns[i][2] for i,num in enumerate(acc)])
    cost  = np.sum([num*patterns[i][3] for i,num in enumerate(acc)])
    print(f"废料：{loss} 接头: {joint} 成本: {cost}")
    print(f"剩余：{left}")
    
 
    print("第二次匹配：")
    left_combination = []
    for i,key in enumerate(L):
        if left[i]>0:
            left_combination += [L[key]] * left[i]
    cost2, combination2 = get_min_cost_combination(left_combination, l, l_min, l_size)
    loss2, joint2 = calc_loss_joint(combination2, l, l_min)
    print(f"废料：{loss2} 接头: {joint2} 成本: {cost2}")
    
    print()
    print("合并数据，依次按如下组合和数量截取：")
    print()   
    # 将最佳方案的组合输出
    # 第一阶段
    for i, num in enumerate(acc):
        if num > 0:
            print(num, '*', patterns[i][-1])
            
    # 第二阶段
    L_keys = list(L.keys())
    L_Values = list(L.values())
    print(1,'*', [L_keys[L_Values.index(num)] for num in combination2])     
    print()   
    print(f"废料长度: {loss+loss2}")
    print(f"接头数量: {joint+joint2}")
    print(f"总成本: {cost+cost2}")
        
decom(l, L)