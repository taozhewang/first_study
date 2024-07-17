import numpy as np
import copy
import time

def solution_initialize(solution_size, need):
    one_solution = np.zeros(np.sum(need), dtype = int)
    start = 0
    for idx, num in enumerate(need):    
        one_solution[start : start + num] = idx
        start += num
    solutions = np.zeros((solution_size, np.sum(need)), dtype = int)
    for i in range(solution_size):
        solution = np.random.permutation(one_solution)
        solutions[i] = solution
    return solutions

def solution_value(solution, l, L, joint, loss):
    waste = 0
    paste = 0
    length = 0
    count = 0
    patterns_id = np.array([], dtype = int)
    left = []
    for idx in solution:
            # idx = int(idx)
            length += L[idx]
            count += 1
            if length >= l + joint:
                paste += 1
                length -= l
                left.extend([length])
            elif length > l:
                paste += 1
                length -= l
                waste += joint - length
                length = joint
                left.extend([l + joint - L[idx], joint])
            elif length == l:
                length = 0
                patterns_id = np.append(patterns_id, np.array([count]))
            elif length > l - joint:
                left.extend([length])
                if l - length <= loss:
                    patterns_id = np.append(patterns_id, np.array([count]))
                waste += l - length
                length = 0
            else:
                if l - length <= loss:
                    patterns_id = np.append(patterns_id, np.array([count]))
                    waste += l - length
                    length = 0
                left.extend([length])

    if length > 0:
        waste += l - length
    return waste, paste, patterns_id, left

# def piece_compare(piece, l, L, joint, loss):

def solution_update(solution, l, L, joint, loss, waste_cost, paste_cost):
    waste, paste, patterns_id, left = solution_value(solution, l, L, joint, loss)
    for _ in range(len(patterns_id)):
        first_id = patterns_id[0]
        last_id = patterns_id[len(patterns_id) - 1]

        piece1 = solution[: first_id]
        piece2 = solution[first_id : last_id]
        piece3 = solution[last_id :]
        
        end_piece = np.append(piece1, piece3)

        new_end_piece = np.random.permutation(end_piece)

        p_waste, p_paste, p_patterns_id, p_left = solution_value(end_piece, l, L, joint, loss)
        p_new_waste, p_new_paste, p_new_patterns_id, p_new_left = solution_value(new_end_piece, l, L, joint, loss)
        # p_new_patterns_id += last_id

        p_cost = waste_cost * p_waste + paste_cost * p_paste + 1 / (len(p_patterns_id) + 1) + 1e-6 / len({p_left})
        p_new_cost = waste_cost * p_new_waste + paste_cost * p_new_paste + 1 / (len(p_new_patterns_id) + 1) + 1e-6 / len({p_new_left})
        if p_new_cost < p_cost:
            solution = copy.deepcopy(np.append(piece2, new_end_piece))
            patterns_id = np.append(patterns_id[1:] - patterns_id[0], p_new_patterns_id + patterns_id[len(patterns_id) - 1] - patterns_id[0])
            waste = waste - p_waste + p_new_waste
            paste = paste - p_paste + p_new_paste
            left[ : len(p_left)]
            # return solution, waste, paste, patterns_id
        else:
            solution = copy.deepcopy(np.append(piece2, end_piece))
            patterns_id = np.append(patterns_id[1:] - patterns_id[0], patterns_id[len(patterns_id) - 1])
        # print(len(solution))
        # print(patterns_id)
    return solution, waste, paste, patterns_id
    
def solution_optimize(solution_size, need, l, L, joint, loss, waste_cost, paste_cost, max_iter, shift_ratio):
    solutions = solution_initialize(solution_size, need)
    
    taboo_list = np.zeros_like(solutions)
    taboo_cost = np.zeros(solution_size)
    for i in range(solution_size):
        taboo_list[i] = solutions[i]
        waste, paste, patterns_id, left = solution_value(solutions[i], l, L, joint, loss)
        taboo_cost[i] = waste_cost * waste + paste_cost * paste + 1 / (len(patterns_id) + 1)
    min_cost = np.min(taboo_cost)
    best_solution = solutions[np.argmin(taboo_cost)]
    taboo_average_cost = np.mean(taboo_cost)

    depth = 0
    while True:

        for i in range(solution_size):
            new_solution, new_waste, new_paste, new_patterns_id = solution_update(solutions[i], l, L, joint, loss, waste_cost, paste_cost)
            solutions[i] = new_solution
            # new_waste, new_paste, new_patterns_id = solution_value(solutions[i], l, L, joint, loss)
            new_cost = new_waste * waste_cost + new_paste * paste_cost + 1 / (len(new_patterns_id) + 1)
            if new_cost < taboo_average_cost:
                replace = True
                for taboo_solution in taboo_list:
                    if np.array_equal(new_solution, taboo_solution):
                        replace = False
                        break
                if replace:
                    taboo_list[i] = solutions[i]
                    taboo_cost[i] = new_cost
            
        new_taboo_average_cost = np.mean(taboo_cost)
        if new_taboo_average_cost >= taboo_average_cost:
                depth += 1
        else: 
            print(f'depth: {depth}')
            print(f'current average cost: {new_taboo_average_cost}')
            print(f'minimum cost: {min_cost}')
            print(f'best solution: {best_solution}')
            depth = 0
        taboo_average_cost = new_taboo_average_cost
        curr_min_cost = np.min(taboo_cost)
            
        if curr_min_cost < min_cost:
            best_solution = taboo_list[np.argmin(taboo_cost)]
            min_cost = curr_min_cost
            
        if depth >= max_iter:
            return best_solution, min_cost
        
        # if depth >= max_iter // 10:
        if depth >= 0:
            taboo_shift_idx = np.argsort(taboo_cost)[ : solution_size // shift_ratio]
            solutions_shift_idx = np.argsort(taboo_cost)[solution_size - solution_size // shift_ratio : ]
            solutions[solutions_shift_idx] = taboo_list[taboo_shift_idx]

fill = input('主动填入数据请按0, 否则按照默认随便按')
if fill == '0':
    l = int(input('原料长度 raw material length: '))
    n = int(input('目标材料种类数 the number of objects: '))
    radius = int(input('形成组合最多允许多少根原材料 radius of the number of raw materials: '))
    losses = int(input('形成组合最多允许多长的余料 max left of patterns: '))

    L = np.zeros(n)
    need = np.zeros(n)
    for i in range(n):
        L[i] = int(input(f'第{i + 1}种目标材料 object L{i + 1}: '))
        need[i] = int(input(f'第{i + 1}种目标材料需要数量 object L{i + 1} need: '))

    joint = int(input('接头最少允许多长: '))
    waste_cost = int(input('余料成本 waste cost: '))
    paste_cost = int(input('粘合成本 paste cost: '))
    
else:
    l = 12000
    L = np.array([4100, 4350, 4700])
    radius = 5
    losses = 200

    need = np.array([852, 658, 162], dtype = int)
    joint = 200
    l_size = 32
    waste_cost = 0.00617 * 2000 * (l_size ** 2) / 1000
    paste_cost = 10
    # waste_cost = 1
    # paste_cost = 1

starttime = time.time()
solution_size = 100
max_iter = 50
shift_ratio = 2

best_solution, min_cost = solution_optimize(solution_size, need, l, L, joint, losses, waste_cost, paste_cost, max_iter, shift_ratio)

print(f'最优解: {best_solution}')
waste, paste, patterns_id = solution_value(best_solution, l, L, joint, losses)
print(f'waste: {waste}')
print(f'paste: {paste}')
print(f'patterns_id: {patterns_id}')

print(f'patterns: {best_solution[0 : patterns_id[0]]}')
for i in range(len(patterns_id) - 1):
    print(f'patterns: {best_solution[patterns_id[i] : patterns_id[i + 1]]}')
if patterns_id[len(patterns_id) - 1] < len(patterns_id):
    print(f'patterns: {best_solution[patterns_id[len(patterns_id) - 1] : ]}')
print(f'cost: {waste * waste_cost + paste * paste_cost}')

endtime = time.time()
print(f'time: {endtime - starttime} s')