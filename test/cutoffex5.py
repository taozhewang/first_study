import numpy as np
import copy
import time
from collections import Counter

def population_initialize(population_size, need):
    one_population = np.zeros(np.sum(need), dtype = int)
    start = 0
    for idx, num in enumerate(need):    
        one_population[start : start + num] = idx
        start += num
    populations = np.zeros((population_size, np.sum(need)), dtype = int)
    for i in range(population_size):
        population = np.random.permutation(one_population)
        populations[i] = population
    return populations

def population_value(population, l, L, joint, loss):
    waste = 0
    paste = 0
    length = 0
    count = 0
    patterns_id = np.array([], dtype = int)
    left = []
    for idx in population:
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
                left.extend([length])
                if l - length <= loss:
                    patterns_id = np.append(patterns_id, np.array([count]))
                    waste += l - length
                    length = 0
                

    if length > 0:
        waste += l - length
    return waste, paste, patterns_id, left

# def piece_compare(piece, l, L, joint, loss):

def population_update(population, l, L, joint, loss, waste_cost, paste_cost):
    waste, paste, patterns_id, left = population_value(population, l, L, joint, loss)
    for _ in range(len(patterns_id)):
        first_id = patterns_id[0]
        last_id = patterns_id[len(patterns_id) - 1]

        piece1 = population[: first_id]
        piece2 = population[first_id : last_id]
        piece3 = population[last_id :]
        
        end_piece = np.append(piece1, piece3)

        new_end_piece = np.random.permutation(end_piece)

        p_waste, p_paste, p_patterns_id, p_left = population_value(end_piece, l, L, joint, loss)
        p_new_waste, p_new_paste, p_new_patterns_id, p_new_left = population_value(new_end_piece, l, L, joint, loss)
        # p_new_patterns_id += last_id

        p_cost = waste_cost * p_waste + paste_cost * p_paste + 1 / (len(p_patterns_id) + 1) + cut * len(set(p_left))
        p_new_cost = waste_cost * p_new_waste + paste_cost * p_new_paste + 1 / (len(p_new_patterns_id) + 1) + cut * len(set(p_new_left))
        if p_new_cost < p_cost:
            population = copy.deepcopy(np.append(piece2, new_end_piece))
            patterns_id = np.append(patterns_id[1:] - patterns_id[0], p_new_patterns_id + patterns_id[len(patterns_id) - 1] - patterns_id[0])
            waste = waste - p_waste + p_new_waste
            paste = paste - p_paste + p_new_paste
            for _ in range(len(p_left)):
                left.pop()
            left.extend(p_new_left)
            # return solution, waste, paste, patterns_id
        else:
            population = copy.deepcopy(np.append(piece2, end_piece))
            patterns_id = np.append(patterns_id[1:] - patterns_id[0], patterns_id[len(patterns_id) - 1])
        # print(len(solution))
        # print(patterns_id)
    return population, waste, paste, patterns_id, left
    
def population_inharitance(populations, population_size, cost):
    parents_index = np.argsort(cost)[: population_size // 2]
    parents = populations[parents_index]
    parents_cost = cost[parents_index]
    offspring = np.zeros_like(populations)
    offspring[: population_size // 2] = parents
    for i in range(population_size // 2):
        success = True
        while success:
            p_index = np.random.choice(parents_index, 2, replace = False, p = parents_cost / np.sum(parents_cost))
            p1 = populations[p_index[0]]
            p2 = populations[p_index[1]]
            for location in range(population_size // 2, -1, -1):
                if Counter(p1[: location + 1]) == Counter(p2[: location + 1]):
                    decsendent = np.append(p1[: location + 1], p2[location + 1 :])
                    offspring[population_size // 2 + i] = decsendent
                    success = False
                    break
                # elif Counter(p1[: location + 1]) == Counter(p2[population_size - location - 1 :]):
                #     decsendent = np.append(p1[: location + 1], p2[: population_size - location - 1])
                #     offspring[population_size // 2 + i] = decsendent
                #     success = False
                #     break
                else:
                    continue 
    return offspring
        
def population_optimize(population_size, need, l, L, joint, loss, waste_cost, paste_cost, max_iter, shift_ratio):
    populations = population_initialize(population_size, need)
    

    cost = np.zeros(population_size)
    for i in range(population_size):
        
        waste, paste, patterns_id, left = population_value(populations[i], l, L, joint, loss)
        cost[i] = waste_cost * waste + paste_cost * paste + 1 / (len(patterns_id) + 1) + cut * len(set(left))
    min_cost = np.min(cost)
    best_population = populations[np.argmin(cost)]
    average_cost = np.mean(cost)

    depth = 0
    unchange = 0
    while True:

        populations = population_inharitance(populations, population_size, cost)
        for i in range(population_size):
            new_population, new_waste, new_paste, new_patterns_id, new_left = population_update(populations[i], l, L, joint, loss, waste_cost, paste_cost)
            populations[i] = new_population
            # new_waste, new_paste, new_patterns_id = solution_value(solutions[i], l, L, joint, loss)
            new_cost = new_waste * waste_cost + new_paste * paste_cost + 1 / (len(new_patterns_id) + 1) + cut * len(set(new_left))
            cost[i] = new_cost

        new_average_cost = np.mean(cost)
        if new_average_cost >= average_cost:
            depth += 1
        else: 
            print(f'depth: {depth}')
            print(f'current average cost: {new_average_cost}')
            print(f'minimum cost: {min_cost}')
            print(f'best population: {best_population}')
            depth = 0
        average_cost = new_average_cost
        curr_min_cost = np.min(cost)
            
        if curr_min_cost < min_cost:
            best_population = populations[np.argmin(cost)]
            min_cost = curr_min_cost
            unchange = 0
        else:
            unchange += 1

        if depth >= max_iter or unchange >= max_unchange:
            return best_population, min_cost
        

fill = input('主动填入数据请按0, 否则按照默认随便按')
if fill == '0':
    l = int(input('原料长度 raw material length: '))
    n = int(input('目标材料种类数 the number of objects: '))
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
    losses = 200

    need = np.array([852, 658, 162], dtype = int)
    joint = 200
    l_size = 32
    waste_cost = 0.00617 * 2000 * (l_size ** 2) / 1000
    paste_cost = 10
    # waste_cost = 1
    # paste_cost = 1

starttime = time.time()
population_size = 100
max_iter = 50
max_unchange = 30
shift_ratio = 2
cut = 1e-2

best_population, min_cost = population_optimize(population_size, need, l, L, joint, losses, waste_cost, paste_cost, max_iter, shift_ratio)

print(f'最优解: {best_population}')
waste, paste, patterns_id, left = population_value(best_population, l, L, joint, losses)
print(f'waste: {waste}')
print(f'paste: {paste}')
print(f'patterns_id: {patterns_id}')

# print(f'patterns: {best_solution[0 : patterns_id[0]]}')

group = []
pat = []
order = []
repeat = []
if 0 not in patterns_id:
    patterns_id = np.append(0, patterns_id)
if len(best_population) not in patterns_id:
    patterns_id = np.append(patterns_id, len(best_population))

for i in range(len(patterns_id) - 1):
    pattern = np.zeros(len(L))
    one_pattern = best_population[patterns_id[i] : patterns_id[i + 1]]
    for idx in one_pattern:
        pattern[idx] += 1
    one_pat = '_'.join(str(x) for x in pattern)
    if one_pat not in pat:
        pat.append(one_pat)
        group.append(pattern)
        order.append(one_pattern)
        repeat.append(1)
    else:
        repeat[pat.index(one_pat)] += 1

for i in range(len(group)):
    print(f'repeated: {repeat[i]}')
    print(f'pattern: {group[i]}')
    print(f'order: {L[order[i]]}')
    print()
for _ in range(repeat[0]):
    final_population = np.array([order[0]])
for i in range(1, len(order)):
    for _ in range(repeat[i]):
        final_population = np.append(final_population, order[i])

waste, paste, patterns_id, left = population_value(best_population, l, L, joint, losses)
print(Counter(left))
for i in left:
    print(i, end = ' ')
print()
# print(f'left: {np.sort
# (left)}')



endtime = time.time()
print(f'time: {endtime - starttime} s')