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

            if patterns_right[i][2] < cut:  
                # 通过cut多少筛选pattern
                patterns_left_plus.pop(origin_index)
                patterns_right_plus.pop(origin_index)
                patterns_left_plus.append(patl)
                patterns_right_plus.append(patterns_right[i])
    patterns = list(zip(patterns_left_plus, patterns_right_plus))
    patterns_main, patterns_property = {}, {}
    for i in range(len(patterns_left_plus)):
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

def patterns_show(pattern, L, l, joint):
    L = np.array(list(L.values()))
    total_length = np.array(L) @ np.array(pattern)
    l_num = total_length // l + bool(total_length % l)
    solution = np.array([])
    for i in range(len(pattern)):
        solution = np.append(solution, np.ones(pattern[i]) * i)
    while True:
        lorder = [[]]
        length = 0
        stage = 0
        order = np.random.permutation(solution)
        
        for idx in order:
            idx = int(idx)
            length += L[idx]

            if length > l + joint:
                lorder[stage].extend([l + L[idx] - length])
                stage += 1
                length -= l
                lorder.extend([[length]])
            elif length > l:
                lorder[stage].extend([l + joint - length, L[idx] - joint])
                stage += 1
                length = joint
                lorder.extend([[joint]])
            elif length == l:
                lorder[stage].extend([L[idx]])
                stage += 1
                length = 0
                lorder.extend([[]])
            elif length > l - joint:
                lorder[stage].extend([l - length])
                stage += 1
                length = 0
                lorder.extend([[]])
            else:
                lorder[stage].extend([L[idx]])
        if lorder[ - 1] == []:
            lorder.pop()
        if len(lorder) == l_num:
            break
    return lorder
    

terms1 = []
terms2 = []
terms4 = []
# 用于组装剩余的几根材料
def calc_left(left, use, paste, accumulate_length, length, stage, joint): 
    # 变量解释：
    # left: 剩余的目标材料
    # use: 已经切下来的目标材料，且是完整的
    # paste: 已经切下来的目标原料，但需要拼接
    # accumulate_length: 已经使用的材料长度（）
    # length: 材料长度分布
    # stage: 已经使用的原材根数
    # joint: 接头长度限制

    # 终止条件：剩余的材料都用完就停下
    if not any(left):
        # 将剩余的原料计入到最终结果中
        clength = copy.deepcopy(length)
        clength[stage].append(l - accumulate_length)
        # terms1用于每根原料中截下的完整的目标材料
        # terms2用于两根原料间拼接的目标材料
        # terms4用于每根原料的截断长度
        terms1.append(use)
        terms2.append(paste)
        terms4.append(clength)
        return 
    # 对每一个目标材料进行遍历
    for i in range(len(left)):
        # 只有在目标材料还有剩余时才进行计算
        if left[i] > 0:
            # 使用copy函数是为了防止改一个全改了的错误
            cleft = copy.deepcopy(left)
            cuse = copy.deepcopy(use)
            cpaste = copy.deepcopy(paste)
            # 当前材料长度caccumulate_length用于记录当前已经使用的材料长度 模 原料l的结果
            # 其大小在 0 到 原料长度l 之间
            caccumulate_length = accumulate_length + L[i]
            # cstage用于记录当前用到的原材根数
            cstage = stage
            # clength用于记录当前原材料中截断长度的分布
            clength = copy.deepcopy(length)
            # 下面分四种情况讨论：
            # 1、当前材料长度大于l+joint
            #       需要将当前目标材料L[i]分成两部分，一部分在上一根原料中，另一部分在下一根原料中
            # 2、当前材料长度大于l，小于等于l+joint
            #       需要先补一份因为joint限制的长度，再将当前目标材料L[i]分成两部分，一部分在上一根原料中，另一部分在下一根原料中
            # 3、当前材料长度大于等于l-joint，小于l
            #       需要将当前目标材料L[i]留在上一根原料中，并补一份因为joint限制的长度余料
            # 4、当前材料长度小于l-joint
            #       直接累加即可
            if caccumulate_length > l + joint:
                caccumulate_length -= l
                cleft[i] -= 1
                cpaste.append(i)
                clength[cstage].append(L[i] - caccumulate_length)
                cstage += 1
                clength.append([caccumulate_length])
                cuse.append([])
                calc_left(cleft, cuse, cpaste, caccumulate_length, clength, cstage, joint)
            elif caccumulate_length > l and caccumulate_length <= l + joint:
                caccumulate_length -= l
                cleft[i] -= 1
                cpaste.append(i)
                clength[cstage].append(joint - caccumulate_length)
                clength[cstage].append(L[i] - joint)
                cstage += 1
                clength.append([joint])
                cuse.append([])
                calc_left(cleft, cuse, cpaste, joint, clength, cstage, joint)
            elif caccumulate_length < l and caccumulate_length > l - joint:
                cleft[i] -= 1
                cuse[cstage].append(i)
                clength[cstage].append(L[i])
                clength[cstage].append( - caccumulate_length + l)
                cstage += 1
                clength.append([])
                cuse.append([])
                cpaste.append(-1)
                calc_left(cleft, cuse, cpaste, 0, clength, cstage, joint)
            else:
                cleft[i] -= 1
                clength[cstage].append(L[i])
                cuse[cstage].append(i)
                calc_left(cleft, cuse, cpaste, caccumulate_length, clength, cstage, joint)

fill = input('主动填入数据请按0, 否则按照默认随便按')
if fill == '0':
    l = int(input('原料长度 raw material length: '))
    n = int(input('目标材料种类数 the number of objects: '))
    L = {}
    for i in range(n):
        L[i] = int(input(f'第{i + 1}种目标材料 object L{i + 1}: '))
    radius = int(input('形成组合最多允许多少根原材料 radius of the number of raw materials: '))
    losses1 = int(input('形成组合最多允许多长的余料 max left of patterns: '))
else:
    l = 12000
    L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
    radius = 10
    losses1 = 0

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
decomposition2(l, L, radius, length, cut, paste, accumulator, path_accumulator, path_left, pointer, stage)
print(patterns_path)

patterns, patterns_main, patterns_property = patterns_simplify(patterns_left, patterns_right)
patterns = patterns_repeated(patterns, patterns_path)
# 会出现pattern1+pattern2=pattern3的情况，虽然说在后续合成过程中不会影响结果，但是会使运算上升一个维度
# 但目前没有很好的处理这样的pattern3的方法
# 已处理：方法如decompostion2()显示
for i, pattern in enumerate(patterns):
    print(i, pattern)

patterns = list(patterns_main.values())
print(patterns)
while True:
    
    k = input('是否使用组合积累到目标? 按回车继续; 按其他键进行下一步 (需要先有积累历史!) accumulate patterns?: press enter to continue or something to exit: ').strip()
    if len(k) > 0:
        break
    n = len(patterns[0])
    need_num = n
    need = np.zeros(n)
    for i in range(n):
        need[i] = int(input(f'第{i + 1}种目标材料需要的数量 how many do you want for the L{i + 1}:'))
    # depth = int(input('how deep do you want to dig?:'))
    propatterns = [[0 for _ in range(len(patterns[0]))]] + patterns
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
            elif currmin == minleft:
                return minaccum, depth

            # depth -= 1
            depth += 1
            # if depth == 0:
            #     return minaccum
            # print('depth:', depth)
            print('curr_stage:', accumulate, currmin)



    # result = calc1(depth)
    result, result_depth = calc1(0)

    print(result[1 :])
    for i in range(1, len(result)):
        if result[i] == 0:
            continue
        print(f'组合 pattern{i - 1}: ', propatterns[i])
        print('用量 use: ', int(result[i]))
        lorder = patterns_show(propatterns[i], L, l, 200)
        for i, j in enumerate(lorder):
            print(f'l ID : {i}; order: {j}')

    print('迭代次数 depth: ', result_depth)
    ac = calcu(result)
    # print(ac)
    print('剩余 left: ', need - ac)



last = need - ac
joint = int(input('接头的最小长度 the min length of the joint: '))
L = list(L.values())

calc_left(last, [[]], [], 0, [[]], 0, joint)

print('每根原料中截下的完整的目标材料：', terms1[0])
print('两根原料间拼接的目标材料：', terms2[0])
print('每根原料的截断长度：', terms4[0])


    
