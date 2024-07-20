# pip install gradio

import csv
import gradio as gr
import numpy as np
import pandas as pd
import time

# 计算成本和组群，计算组别 
def evaluate(combinations, l_size, m_size, d_size, l_price, joint_price):
    '''
        combinations:禁忌表
        l_size:钢筋长度
        m_size:最小长度
        d_size：钢筋直径
        l_price:钢筋价格
        joint_price:接头价格
    '''
    combinations_size = len(combinations)                           # 禁忌表大小
    _l = np.ones(combinations_size)*l_size                               # 剩余长度
    group_count = np.zeros(combinations_size,dtype=int)             # 小组个数
    group_firstpos = np.zeros(combinations_size,dtype=int)          # 第一个小组的位置
    group_endpos = np.zeros(combinations_size,dtype=int)            # 最后一个小组的位置
    loss = np.zeros(combinations_size,dtype=float)                  # 余料
    joint = np.zeros(combinations_size,dtype=int)                   # 接头
    cost = np.zeros(combinations_size,dtype=float)                  # 成本

    for i in range(len(combinations[0])): 
        while True:
            idxs=np.where(_l<combinations[:,i])[0]
            if len(idxs)==0: break
            _l[idxs] += l_size
            joint[idxs] += 1
        _l -= combinations[:,i]

        # 是否存在左边接头长度不够的情况
        min_idx = np.where((l_size-_l)<m_size)[0]
        if len(min_idx)>0:
            add_len = m_size - (l_size-_l)
            loss[min_idx] += add_len[min_idx]
            _l[min_idx] -= add_len[min_idx]

        # 确定第一个小组的最后一个位置,如果第一个位置为0且有废料，则将其作为第一个位置
        fidx = np.where((group_firstpos==0) & (_l<m_size))[0]
        if len(fidx)>0:
            group_firstpos[fidx]= i+1

        # 确定其他
        idxs=np.where(_l<m_size)[0]
        if len(idxs)>0:
            loss[idxs] += _l[idxs]
            group_count[idxs] += 1
            group_endpos[idxs] = i+1
            _l[idxs] = l_size

    loss += _l

    cost_param = (d_size**2)*0.00617*l_price/1000
    cost = loss*cost_param + joint*joint_price
    return loss, joint, cost, group_count, group_firstpos, group_endpos

# 比较两个group，返回接头数量选择较小的group
def get_best_solution_group(group1, group2, l_size, m_size):
    joint1 = joint2 = 0

    _l = l_size
    for length in group1:
        while _l < length:
            _l += l_size
            joint1 += 1
        _l -= length
        if l_size-_l<m_size: _l -= m_size-(l_size-_l)
        if _l < m_size:
            _l = l_size

    _l = l_size
    for length in group2:
        while _l < length:
            _l += l_size
            joint2 += 1
        _l -= length
        if l_size-_l<m_size: _l -= m_size-(l_size-_l)
        if _l < m_size:
            _l = l_size

    return group1 if joint1 < joint2 else group2

# 获得邻域解
# 选择第一组，打乱顺序加入到最后一组后面，期望找到小的接头数组合；同时打乱最后一组剩下的长度的顺序，期望可以发现新的组合
def get_neighbor(combinations, group_firstpos, group_endpos, l_size, m_size):
    combinations_size = len(combinations)
    combinations_length = len(combinations[0])
    for i in range(combinations_size):
        combination, firstpos, endpos = combinations[i], group_firstpos[i], group_endpos[i]
        if endpos < combinations_length-1:  
            first_group = get_best_solution_group(combination[:firstpos], np.random.permutation(combination[:firstpos]), l_size, m_size)
            combinations[i] = np.concatenate((combination[firstpos:endpos], first_group, np.random.permutation(combination[endpos:])))
        else:                                  # 最后一组的位置刚好在最后
            combinations[i] = np.concatenate((combination[firstpos:], np.random.permutation(combination[:firstpos])))
    return combinations

# 计算不同组的个数
def get_distinct_group_count(best_solution, l_size, m_size):
    group_count = 0
    _l=l_size
    _group=[]
    counters=[]    
    L_values = list(set(best_solution))
    for p in best_solution:
        _group.append(p)        
        while _l<p:
            _l+=l_size
        _l -= p

        if _l<m_size:
            # 统计计数
            _counter=[0 for _ in range(len(L_values))]
            for x in _group:
                _counter[L_values.index(x)]+=1
                
            if _counter not in counters:
                group_count+=1
            else:    
                counters.append(_counter)

            _l=l_size
            
    if len(_group)>0:
        _counter=[0 for _ in range(len(L_values))]
        for x in _group:
            _counter[L_values.index(x)]+=1        
        if _counter not in counters:
            group_count+=1
                    
    return group_count        
# 分解最佳方案
def get_data(best_solution, l_size, m_size):
    counters=[]
    groups=[]
    nums=[]
    loss=[]
    joints=[]
    used=[]
    _group=[]
    _l=l_size
    L_values = list(set(best_solution))
    _joint = 0
    _used = 1
    for p in best_solution:
        _group.append(int(p))        
        while _l<p:
            _l+=l_size
            _joint+=1
            _used+=1
        _l -= p

        if _l<m_size:
            # 统计计数
            _counter=[0 for _ in range(len(L_values))]
            for x in _group:
                _counter[L_values.index(x)]+=1
                
            if _counter in counters:
                nums[counters.index(_counter)]+=1
            else:    
                counters.append(_counter)
                groups.append(_group.copy())
                nums.append(1)
                loss.append(_l)
                joints.append(_joint)
                used.append(_used)
            _group=[]
            _l=l_size
            _joint=0
            _used=1
            
    if len(_group)>0:
        _counter=[0 for _ in range(len(L_values))]
        for x in _group:
            _counter[L_values.index(x)]+=1
        counters.append(_counter)
        groups.append(_group.copy())
        nums.append(1)
        loss.append(_l)
        joints.append(_joint)
        used.append(_used)
    return groups, nums, loss, joints, used

# 禁忌搜索算法
def predict(l_size,m_size,d_size,l_price,joint_price,need):
    print(l_size,m_size,d_size,l_price,joint_price,need)   
    # 最大循环次数
    max_iterations = 1000000
    # 禁忌表大小
    tabu_size = 100
    # 最大停滞次数
    max_stagnation = 500    
    
    # 图表
    plot_y=[]
    plot_x=[]
    start_time = time.time()
    
    base_combination = []
    for size,num in need:
        try:
            base_combination += [int(size)] * int(num)
        except:
            continue

    # 采用随机初始解
    tabu_list = np.array([np.random.permutation(base_combination) for _ in range(tabu_size)])
    # 计算初始解的评估
    tabu_loss, tabu_joint, tabu_cost, tabu_group_count, group_firstpos, group_endpos = evaluate(tabu_list, l_size, m_size, d_size, l_price, joint_price)
    # 记录最佳解
    best_solution = None
    # 记录最佳解的评估
    best_cost = np.inf
    best_loss = 0
    best_joints = 0
    # 记录连续没有改进的次数
    nochange_count = 0

    for i in range(max_iterations):
        # 从禁忌表中获得一组邻域解
        neighbors = get_neighbor(tabu_list, group_firstpos, group_endpos, l_size, m_size)
        # 计算邻域解的评估
        neighbors_loss, neighbors_joint, neighbors_cost, neighbors_group_count, group_firstpos, group_endpos = evaluate(neighbors, l_size, m_size, d_size, l_price, joint_price)
        
        # 选择最佳邻域解
        best_idx = np.argmin(neighbors_cost)
        best_neighbor_cost = neighbors_cost[best_idx] 
                      
        # 禁忌搜索
        # 如果邻域解比最佳解好，更新最佳解
        if best_neighbor_cost < best_cost:
            best_solution = np.copy(neighbors[best_idx])
            best_cost = best_neighbor_cost
            nochange_count = 0
            best_loss = neighbors_loss[best_idx]
            best_joints = neighbors_joint[best_idx]
        elif best_neighbor_cost == best_cost:
            old_group_count = get_distinct_group_count(best_solution, l_size, m_size)
            new_group_count = get_distinct_group_count(neighbors[best_idx], l_size, m_size)
            if new_group_count < old_group_count:
                best_solution = np.copy(neighbors[best_idx]) 
                nochange_count = 0           
                        
        plot_y.append(best_cost)
        plot_x.append(time.time()-start_time)

        nochange_count += 1

        # 如果邻域解比当前解好，则更新禁忌组
        update_count = 0
        avg_waste = np.average(tabu_cost)
        avg_groups_count=np.average(tabu_group_count)
        for idx, waste in enumerate(neighbors_cost):
            if (neighbors_group_count[idx]>avg_groups_count) or (waste < avg_waste):
                update_count += 1
                tabu_list[idx]=neighbors[idx]                
                tabu_cost[idx]=waste
                tabu_group_count[idx] = neighbors_group_count[idx]       

        if i % 10 == 0:
            groups_copunt=np.average(tabu_group_count)
            print(f"{i}: 禁忌组平均组个数:{groups_copunt}, 最佳成本:{best_cost}, 余料: {best_loss} 接头: {best_joints} 停滞次数: {nochange_count}/{max_stagnation}")

            # 如果连续 max_stagnation 次没有改进，则退出循环
            if nochange_count>max_stagnation:
                print("已达到目标，退出循环")
                break                            
            
            plot = gr.LinePlot(
                    value=pd.DataFrame({"x": plot_x, "y": plot_y}),
                    x="x",
                    y="y",
                    y_lim=[min(plot_y)-100,max(plot_y)+100],
                    x_title="时间(s)",
                    y_title="成本",
                    title="成本收敛图",
                    height=350,
                )

            yield [best_cost, plot, None, None]
    groups, nums, loss, joints, used = get_data(best_solution, l_size, m_size)  
    data = []
    for i in range(len(groups)):
        data.append([nums[i], groups[i], used[i], joints[i], loss[i]])
    data = sorted(data, key=lambda x:x[0], reverse=True)
                
    filename= "output.csv"
    with open(filename, 'w', newline='', encoding="GBK") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["截取次数", "截取顺序" ,"耗用钢筋", "接头数", "余料"])
        writer.writerows(data)
        
    csvdownload = gr.File(value=filename, visible=True)
    yield [best_cost, plot, data, csvdownload]


if __name__ == "__main__":
    with gr.Blocks() as web:
        with gr.Row():
            gr.Markdown("""# 钢筋断料计算""")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**输入：**")
            with gr.Column():
                gr.Markdown("**输出：**")
        with gr.Row():
            with gr.Column():
                l_size = gr.Number(label='原始钢筋的长度(mm)', value=12000, minimum=1000, maximum=50000)
                m_size = gr.Number(label='最小可接长度(mm)', info='小于此数值（不含）的部分会做为废料', value=200, minimum=10, maximum=500)
                d_size = gr.Radio([6,8,10,12,16,20,25,32,40,50], label='钢筋的直径(mm)', value=32)
                l_price = gr.Number(label='钢筋价格(元/吨)', info='成本计算公式为(单位mm)： (直径^2*0.00617)*(废料长度/1000)*单价', value=2000, minimum=1500, maximum=3000)  
                joint_price = gr.Number(label='接头价格(元/个)', info='价格已经包含了人力成本，计算公式为： 接头个数*接头单价', value=10, minimum=1, maximum=100)  
                need = gr.Dataframe(
                    headers=["输出的钢筋长度(mm)", "输出的钢筋根数"],
                    col_count=(2, "fixed"),
                    datatype="number",
                    type="array",
                    label="下面是需要输出的钢筋长度和钢筋根数",
                    value = [[4100, 852], [4350, 658], [4700, 162]],
                )      
                button = gr.Button(value="开始计算")               
            with gr.Column():
                amount = gr.Number(label='成本(元)')  
                out = gr.Dataframe(
                    headers=["截取次数", "截取顺序" ,"耗用钢筋", "接头数", "余料"],
                    type="pandas",
                    datatype=["number", "str","number", "number","number"],  
                    interactive = False,
                    wrap = True, 
                ) 
                csvdownload = gr.File(interactive=False, visible=False)               
                plot = gr.LinePlot(show_label=False,
                                   height=350,                                 
                                   )
        button.click(predict, inputs=[l_size, m_size, d_size, l_price, joint_price, need] , outputs=[amount, plot, out, csvdownload])    
    web.launch()

