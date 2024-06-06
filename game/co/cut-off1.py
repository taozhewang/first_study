import numpy as np
import copy
# 不适合L中带有特别大的Li的情况（Li几乎等于原料）

def decomposition2(l, L, n, length, cut, paste, accumulator, path_accumulator, path_left, pointer, stage):
    L_length = list(L.values())
    if pointer == len(L_length) - 1: # 如果当前pointer已经指向了最后一种L，那么不再移动pointer
        if stage == n:               # 如果pattern总长度已经超过了n（也就是radius）倍的原长，则停止计算
            if len(path_accumulator):   

                print(path_accumulator)           # 在这里统计同一个路径上产生的大pattern的种类

                for pattern in path_accumulator:    # path_accumulator里的第一个元素有可能是小pattern
                    if pattern not in patterns_path:    # 对于是大pattern的情况，会有另外的path_accumulator统计到
                        patterns_path.append(pattern)
            return
        Re_accumulator = copy.deepcopy(accumulator)    
        Re_path_accumulator = copy.deepcopy(path_accumulator)
        Re_path_left = copy.deepcopy(path_left)

        length += L_length[pointer]
        Re_accumulator[pointer] += 1
        if length > l:          # 采用的是首尾相接的办法
            length = length - l # 将length限制在0<=length<l间
            stage += 1          # 用stage表示已经用完的l数目
            cut += 1            # 只有在length正好到0的时候才不会让cut增加
            paste += 1          # paste同上
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
        elif length == l:
            length = 0
            patterns_left.append(Re_accumulator)
            patterns_right.append([0, stage + 1, cut, paste])
            if 0 in Re_path_left:
                Re_path_accumulator.append(Re_accumulator)
            Re_path_left.append(0)
            stage += 1
            pointer = 0 # 在left为0的情况下将pointer重置到0是为了在原有pattern上产生新的pattern
                        # 在后面需要排除可以被两个独立pattern组成的大pattern的情况
                        # 为了做到这一点需要统计可以被拆成两个独立pattern的大pattern
                        # 所以统计在每一个pattern产生过程中是否有小pattern产生
                        # 如果不重置pointer，大的pattern的产生路径中可能不会产生小pattern
                        # 因为排序的不同会影响大pattern能否被拆出小pattern
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
        else:
            # 只有在pattern总长正好抵达l的整数倍时，才能节省一次断料cut
            cut += 1
            left = l - length
            if left <= losses1:
                # 在当前合成长度与原料截断处的距离left小于losses1时，
                # 记录下当前的pattern组成、原料个数stage、断料次数cut、接头用量paste
                patterns_left.append(Re_accumulator)
                patterns_right.append([left, stage + 1, cut, paste])
                Re_path_left.append(left)
                Re_path_accumulator.append(Re_accumulator)
                

                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
            else:
                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
    else:
        if stage == n:
            if len(path_accumulator):

                print(path_accumulator)

                for pattern in path_accumulator:
                    if pattern not in patterns_path:
                        patterns_path.append(pattern)
            return
        Re_accumulator = copy.deepcopy(accumulator)
        Re_path_accumulator = copy.deepcopy(path_accumulator)
        Re_path_left = copy.deepcopy(path_left)
        decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer + 1, stage)

        length += L_length[pointer]
        Re_accumulator[pointer] += 1
        if length > l:
            length = length - l
            stage += 1
            cut += 1
            paste += 1
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
        elif length == l:
            length = 0
            patterns_left.append(Re_accumulator)
            patterns_right.append([0, stage + 1, cut, paste])
            if 0 in Re_path_left:
                Re_path_accumulator.append(Re_accumulator)
            Re_path_left.append(0)
            stage += 1
            pointer = 0
            decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
        else:
            cut += 1
            left = l - length
            if left <= losses1:
                patterns_left.append(Re_accumulator)
                patterns_right.append([left, stage + 1, cut, paste])
                Re_path_left.append(left)
                Re_path_accumulator.append(Re_accumulator)
                

                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)
            else:
                decomposition2(l, L, n, length, cut, paste, Re_accumulator, Re_path_accumulator, Re_path_left, pointer, stage)


        
def patterns_simplify(patterns_left, patterns_right):
    # 用来缩减重复pattern的数量，同时去除同样的pattern但cut，paste更多的pattern
    # 同时将产生的排序不同但组成相同的pattern简化到一种作为代表
    patterns_left_plus, patterns_right_plus = [], []
    k = len(patterns_left)
    for i in range(k):
        patl = patterns_left[i]
        if patl not in patterns_left_plus:
            patterns_left_plus.append(patl)
            patterns_right_plus.append(patterns_right[i])
        else:
            origin_index = patterns_left_plus.index(patl)
            cut = patterns_right_plus[origin_index][2]

            if patterns_right[i][2] < cut:  # 通过cut多少筛选pattern
                patterns_left_plus.pop(origin_index)
                patterns_right_plus.pop(origin_index)
                patterns_left_plus.append(patl)
                patterns_right_plus.append(patterns_right[i])
    patterns = list(zip(patterns_left_plus, patterns_right_plus))
    patterns_main, patterns_property = {}, {}
    for i in range(len(patterns_left_plus)):
        # patterns_main[i] = [patterns_left_plus[i], patterns_right_plus[i][0]]
        patterns_main[i] = patterns_left_plus[i]
        patterns_property[i] = patterns_right_plus[i]
    return patterns, patterns_main, patterns_property

def patterns_repeated(patterns, patterns_path):
    p = []
    for i in patterns:
        p.append(i[0])
    for paths in patterns_path:
        loc = p.index(paths)
        patterns.pop(loc)
        p.pop(loc)
    return patterns

def patterns_decomposition(pattern, l, L, joint, length, count, pointer, stage):
# 通过遍历寻找pattern的不同组成方法
# 加入了joint来约束余料长度，顺便剪枝

        p_length = len(pattern)
        L_length = list(L.values())
        Re_pattern = copy.deepcopy(pattern)
        Re_count = copy.deepcopy(count)

        # if len(accumulator) > max_way:
        #     return

        if Re_pattern[pointer] == 0:
            pointer = (pointer + 1) % p_length
        else:
            length += L_length[pointer]
            

            if length > l:
                length -= l
                a = Re_count[stage]

                left = L_length[pointer] - length
                if left < joint or length < joint:
                    return
                
                a.append(left)
                Re_count[stage] = a
                stage += 1
                Re_count[stage] = [length]
                Re_pattern[pointer] = Re_pattern[pointer] - 1

                if not any(Re_pattern):
                    if l == length:
                        Re_count.pop(stage)
                        accumulator.append(Re_count)
                        return
                    else:
                        left = l - length
                        a = Re_count[stage]
                        a.append(left)
                        Re_count[stage] = a
                        accumulator.append(Re_count)

                        # print(Re_count)

                        return

                                     
                else:
                    for step in range(p_length):
                        Re_pointer = step
                        if Re_pattern[Re_pointer] != 0:
                            patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)

            elif length == l:
                length = 0
                a = Re_count[stage]
                a.append(L_length[pointer])
                Re_count[stage] = a
                stage += 1
                Re_count[stage] = []
                Re_pattern[pointer] = Re_pattern[pointer] - 1

                if not any(Re_pattern):
                    Re_count.pop(stage)
                    accumulator.append(Re_count)

                    # print(Re_count)

                    return
                    
                else:
                    for step in range(p_length):
                        Re_pointer = step
                        if Re_pattern[Re_pointer] != 0:
                            patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)

            else:
                a = Re_count[stage]
                a.append(L_length[pointer])
                Re_count[stage] = a
                Re_pattern[pointer] = Re_pattern[pointer] - 1

                if not any(Re_pattern):
                    left = l - length
                    a = Re_count[stage]
                    a.append(left)
                    Re_count[stage] = a
                    accumulator.append(Re_count)

                    # print(Re_count)

                    return
            
                else:
                    for step in range(p_length):
                        Re_pointer = step
                        if Re_pattern[Re_pointer] != 0:
                            patterns_decomposition(Re_pattern, l, L, joint, length, Re_count, Re_pointer, stage)
                            
def patterns_decomposition_summon(pattern, l, L, joint):
    p_length = len(pattern)
    for i in range(p_length):
        if pattern[i] != 0:
            patterns_decomposition(pattern, l, L, joint, 0, {0: []}, i, 0)
    return

print("Enter the parameters below, press Enter to use the default value.")
l = int(input('raw material length [12000]: ') or 12000)
n = int(input('the number of objects [3]: ') or 3)
L = {}
L_D = [4100, 4350, 4700]
for i in range(n):
    if i<3:
        L[f"L{i+1}"] = int(input(f'object L{i + 1} [{L_D[i]}]: ') or L_D[i])    
    else:
        L[f"L{i+1}"] = int(input(f'object L{i + 1} []: '))
radius = int(input('radius of the number of raw materials [10]: ') or 10)
losses1 = int(input('max left of patterns [0]: ') or 0)

patterns_left = []
patterns_right = []
patterns_path = []

cut = 0 
paste = 0
accumulator = [0 for _ in L.keys()]
pointer = 0
stage = 0
length = 0
path_accumulator = []
path_left= []

# 获得所有符合条件的组合
decomposition2(l, L, radius, length, cut, paste, accumulator, path_accumulator, path_left, pointer, stage)
#print(patterns_path)

# 精简组合，取得同组合最低的成本
patterns, patterns_main, patterns_property = patterns_simplify(patterns_left, patterns_right)

# 删除掉子组合
patterns = patterns_repeated(patterns, patterns_path)
# 会出现pattern1+pattern2=pattern3的情况，虽然说在后续合成过程中不会影响结果，但是会使运算上升一个维度
# 但目前没有很好的处理这样的pattern3的方法
# 已处理：方法如decompostion2()显示
print("所有可用组合如下：")
print("序号 [组合]    [余料长度，消耗原料根数，切割次数，接头个数]")

for i, pattern in enumerate(patterns):
    print(i, pattern)

patterns = list(patterns_main.values())
# print(patterns)

while True:
    
    k = input('accumulate patterns[Y]?: press Enter to accumulate. Ctrl+C to exit.') or 'Y'
    if k.upper() != 'Y':
        break
    n = len(patterns[0])
    need_num = n
    need = np.zeros(n)
    d_v = [552, 658, 462]
    for i in range(n):
        if i<3:
            need[i] = int(input(f'how many do you want for the L{i + 1} [{d_v[i]}]:') or d_v[i])
        else:
            need[i] = int(input(f'how many do you want for the L{i + 1}:'))
            
    depth = int(input('how deep do you want to dig? [100]:') or 100)
    # 增加一个 [0,0,0] ?
    propatterns = [[0 for _ in range(len(patterns[0]))]] + patterns
    
    # 计算耗用
    def calcu(accumulate):
        lvalues = L.values()
        count = np.zeros(len(lvalues))
        for i, num in enumerate(accumulate):
            count += np.array(propatterns[i]) * num
        return count

    def calc1(depth):
        
        accumulate = np.zeros(len(patterns) + 1) # the first space is used for doing nothing
        backward_forbid = []
        forward_forbid = []
        forbid_length = 10
        maxleft = np.sum(need ** 2) + 1
        minleft = maxleft
        
        # decrease then increase
        while True:
            print('////////////////////////////////')
            cross = np.ones((len(propatterns), len(propatterns))) * maxleft # collect all potential left
            for i in range(len(propatterns)):
                accumulate[0] = 50
                if accumulate[i] > 0:
                    Re_accumulate = copy.deepcopy(accumulate)
                    Re_accumulate[i] -= 1
                    for j in range(len(propatterns)):
                        Re_Re_accumulate = copy.deepcopy(Re_accumulate)
                        Re_Re_accumulate[j] += 1
                        left_list = need - calcu(Re_Re_accumulate)
                        left = np.sum(left_list ** 2)

                        # print('current issue:', (i, j), left)
                        if all(left_list >= 0):
                            if left < minleft:
                                cross[i, j] = left
                            # elif i not in forward_forbid or j not in backward_forbid:
                            elif (i, j) not in list(zip(forward_forbid, backward_forbid)):
                                cross[i, j] = left
                            else:
                                cross[i, j] = maxleft
                        else:
                            cross[i, j] = maxleft
            for i in range(len(propatterns)):
                cross[i, i] = maxleft
            # print(cross)

            currmin = np.min(cross)
            loc_list = list(np.where(cross == currmin))
            loc = list(zip(loc_list[0], loc_list[1]))
            x, y = loc[0]
            accumulate[x] -= 1
            accumulate[y] += 1
            
            forward_forbid.append(x)
            backward_forbid.append(y)
            if len(forward_forbid) > forbid_length:
                forward_forbid.pop(0)
            if len(backward_forbid) > forbid_length:
                backward_forbid.pop(0)

            if currmin < minleft:
                minleft = currmin
                minaccum = copy.deepcopy(accumulate)

            depth -= 1
            if depth == 0:
                return minaccum
            print('depth:', depth)
            print('curr_stage:', accumulate, currmin)

    
        # 计算成本
    def calccost(accumulate, need):
        print(f"calc cost:")
        left_sum = 0
        paste_sum = 0
        lost = copy.deepcopy(need)
        for i, num in enumerate(accumulate):
            if i==0: continue
            print(i, num, propatterns[i], patterns_property[i-1])
            lost -= np.array(propatterns[i]) * num
            left_sum += np.sum(patterns_property[i-1][0]) * num
            paste_sum += np.sum(patterns_property[i-1][3]) * num
        _l=l 
        L_Values= list(L.values())
        print(f"left: {left_sum} paste: {paste_sum}")        
        print(f"continue calc lost cost: {lost}")
        for i, num in enumerate(lost):            
            while num > 0:
                while _l<L_Values[i]:
                    _l+=l
                    paste_sum+=1
                _l -= L_Values[i]
                num -= 1
                if _l<200:
                    left_sum+=_l
                    _l=l        
                    
        left_sum+=_l%l
        print(f"totel left: {left_sum} totel paste: {paste_sum}")
        cost_left_param = 0.00617 * 2000 * (32 ** 2) / 1000
        cost = left_sum * cost_left_param + paste_sum * 10
        return cost
                

    result = calc1(depth)

    print(result)
    ac = calcu(result)
    print(ac)
    print(need - ac)
    # 计算余料和接头
    cost = calccost(result, need)
    print(f"总成本为: {cost}")
    
