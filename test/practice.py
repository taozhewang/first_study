# Q1: Given a list of interger and a target,
#       find the two interger that add to the target and return their index
# def find_index(nums, target):
#     nums = list(nums)
#     for i in range(len(nums)):
#         for j in range(i + 1, len(nums)):
#             if target == nums[i] + nums[j]:
#                 return [i, j]

# a = find_index([3, 3], 6)
# print(a)


# Q2: Given three numbers, find that whether they can form a triangle
# def in_order(nums):
#     for i in range(len(nums) - 1):
#         for j in range(len(nums) - 1 - i):
#             if nums[j] > nums[j + 1]:
#                 nums[j], nums[j + 1] = nums[j + 1], nums[j]
#     return nums
# def triangle(nums):
#     nums = in_order(nums)
#     if len(nums) != 3:
#         return False
#     if nums[0] == nums[1] == nums[2]:
#         return 'equilateral'
#     elif nums[0] + nums[1] < nums[2]:
#         return None
#     elif nums[0] == nums[1] or nums[1] == nums[2]:
#         return 'isosceles'
#     else:
#         return 'scalene'

# print(triangle([3, 3, 3]))
# print(triangle([3, 4, 5]))
# print(triangle([13, 8, 3]))
# print(triangle([9, 5, 5]))


# Q?1:best marriage
# from <<college Admissions and the Stability of the Marriage>>
# Assume1: there are only males and females.
# Assume2: all ladies and gentlemen are heterosex.
# Assume3: the number of males and females are equal.
# Assume4: all of the ladies and gentlemen will get marriage.

# import copy
# def male_find_marriage(male, female):
#     # male and female are dictionaries, their keys are themselves, and their values are their love ones.
#     '''examples:
#     >>> male = {'Bob': ['Alice', 'Catheline'], 'David': ['Catheline', 'Alice']}
#     >>> female = {'Alice': ['David', 'Bob'], 'Catheline': ['David', 'Bob']}
#     >>> a = male_find_marriage(male, female)
#     >>> print(a)
#     [['Alice', 'Bob'], ['Catheline', 'David']]
#     '''
#         # Step 1: male choose their best love.
#         # First create a dictionary that represent the contemporal relationship.
#         # initialize the dictionary
#     final_marriage = []
#     while male != {}:
#         curr_marriage_male = {}
#         curr_marriage_female = {}
#         for woman in female.keys():
#             curr_marriage_female[woman] = []
#         for man in male.keys():
#             for woman in male[man]:
#                 if woman in female:
#                     curr_marriage_female[woman].append(man)
#                     break
#         # Step 2: female choose their best love, from males who pursue them.
#         # for the same female, return the rank of their pursuers.
#         fefefefefe = copy.deepcopy(female)
#         for woman in fefefefefe.keys():
#             for man in female[woman]:
#                 if man in curr_marriage_female[woman]:
#                     final_marriage.append([woman, man])
#                     male.pop(man)
#                     female.pop(woman)
#                     break
#     return final_marriage

# male = {'A': [1, 2, 3], 'B': [3, 1, 2], 'C': [2, 3, 1]}
# female = {1: ['C', 'A', 'B'], 2: ['B', 'C', 'A'], 3: ['A', 'B', 'C']}
# a = male_find_marriage(male, female)
# print(a)

# male = {'A': [4, 3, 1, 2], 'B': [2, 4, 1, 3], 'C': [4, 1, 2, 3], 'D': [3, 2, 1, 4]}
# female = {1: ['A', 'B', 'C', 'D'], 2: ['A', 'D', 'C', 'B'], 3: ['B', 'A', 'C', 'D'], 4: ['D', 'B', 'C', 'A']}
# a = male_find_marriage(male, female)
# print(a)
# male = {'A': [4, 3, 1, 2], 'B': [2, 4, 1, 3], 'C': [4, 1, 2, 3], 'D': [3, 2, 1, 4]}
# female = {1: ['A', 'B', 'C', 'D'], 2: ['A', 'D', 'C', 'B'], 3: ['B', 'A', 'C', 'D'], 4: ['D', 'B', 'C', 'A']}
# b = male_find_marriage(female, male)
# print(b)


# # Q2?: create a sudoku game
# import numpy as np
# import random
# def sudoku_building(sudoku, x, y):
#     available_numbers = []
#     for i in range(9):
#         n = available(sudoku, i + 1, x, y)
#         if n:
#             available_numbers.append(n)
#     if available_numbers == []:
#         return False, sudoku
#     number = random.choice(available_numbers)
#     sudoku[y][x] = number
#     return True, sudoku

# def sudoku_create():
#     sudoku = np.zeros((9, 9), dtype=np.uint32)
#     for y in range(9):
#         for x in range(9):
#             success, sudoku = sudoku_building(sudoku, x, y)
#             if not success:
#                 return success, sudoku
#     return success, sudoku

# def available(sudoku, i, x, y):
#     smaller = np.array([[sudoku[yy][xx] for yy in range(3*(y//3), 3*(1+y//3))]
#                          for xx in range(3*(x//3), 3*(1+x//3))])
#     if i in sudoku[y] or i in sudoku.T[x] or i in smaller:
#         return False
#     else:
#         return i

# def game_forming(sudoku):
#     n = 40
#     i = 0
#     while i < n:
#         x = random.randint(0,8)
#         y = random.randint(0,8)
#         if sudoku[y, x] != 0:
#             sudoku[y, x] = 0
#             i += 1
#     return sudoku

# success = False
# while not success:
#     success, sudoku = sudoku_create()
# print(sudoku)
# print(game_forming(sudoku))

# from math import sqrt
# def bi_decomposition(num, size = 50):
#     if num >= 1 or num < 0:
#         return False
#     elif num == 0:
#         return 0
#     else:
#         start = 0
#         yes = []
#         for i in range(size):
#             n = start + 2 ** (- 1 - i)
#             if n == num:
#                 return n
#             elif n < num:
#                 start = n
#                 # yes.append(i)
#         return (num - start) / num
# print(bi_decomposition(sqrt(2) - 1))


# # Hamiton
# import numpy as np
# import copy
# '''
# distance: matrix
# ------A--B--C--D--E
#     A 0  3  2  2  3
#     B 3  0  2  3  3  
#     C 2  2  0  2  3
#     D 2  3  3  0  2
#     E 3  3  3  2  0
# '''
# distance = np.array([[0, 3, 4, 5, 6, 5, 4],
#                      [3, 0, 9, 8, 7, 4, 3],
#                      [4, 9, 0, 1, 1, 3, 2],
#                      [5, 8, 1, 0, 2, 2, 1],
#                      [6, 7, 1, 2, 0, 1, 2],
#                      [5, 4, 3, 2, 1, 0, 9],
#                      [4, 3, 2, 1, 2, 9, 0]])

# def d(name1, name2):
#     indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
#     x, y = indices[name1], indices[name2]
#     return distance[x, y]

# def path_d(path):
#     distance = 0
#     for i in range(len(path) - 1):
#         m, n = path[i], path[i + 1]
#         distance += d(m, n)
#     return distance

# def swap_path(path, i, j):
#     newpath = copy.deepcopy(path)
#     newpath[i], newpath[j] = path[j], path[i]
#     return newpath

# def forbidsol(p0):
#     forbid_length = 4
#     forbid_group = []
#     total_d = np.sum(distance)
#     # print(total_d)

#     path = p0
#     depth = 10
#     while True:
#         print('current depth:', depth)
#         print('current path:', path)
        
#         depth -= 1
#         curr_d = path_d(path)
#         paths = []
#         ds = []
#         swaps = []

#         for i in range(1, len(path) - 2):
#             for j in range(i + 1, len(path) - 1):
#                 print('/', end = '')
#                 newpath = swap_path(path, i, j)
                
#                 new_d = path_d(newpath)
                
#                 # print(forbid_group)

#                 if [i, j] not in forbid_group or new_d < total_d:

#                     print('new path:', newpath)
#                     print('new distance:', new_d)

#                     paths.append(newpath)
#                     ds.append(new_d)
#                     swaps.append([i, j])
#         min_d = min(ds)
#         loc = ds.index(min_d)
#         min_path = paths[loc]
        
#         forbid_group.append(swaps[loc])
#         if len(forbid_group) == forbid_length:
#             forbid_group.pop(0)
        
#         if min_d < total_d:
#             total_d = min_d
#             total_path = min_path
        
#         if depth == 0:
#             return total_path, total_d
        
#         path = min_path
#         # print()
#         # print(paths)
#         # print(total_path, total_d)
# p0 = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A'])
# psolution, dsolution = forbidsol(p0)
# print()
# print('solution:', psolution, dsolution)

# # transformation on C
# import numpy as np
# import plotly.graph_objects as go
# def print_Im(f, leftbound, rightbound, lowerbound, upperbound, step = 0.5):
#     fig = go.Figure()
#     Dfx = []
#     Dfy = []
#     Imx = []
#     Imy = []
#     for i in np.arange(int(leftbound), int(rightbound) + 1, step):
#         # print(i)
#         for j in np.arange(int(lowerbound), int(upperbound) + 1, step):
#             # print(i)
#             Dfx.append(i)
#             Dfy.append(j)
#             x, y = f(i, j)
#             Imx.append(x)
#             Imy.append(y)

#     fig.add_scatter(x = Dfx, y = Dfy, mode = 'lines+markers', name = 'Definitive Field')
#     fig.add_scatter(x = Imx, y = Imy, mode = 'lines+markers', name = 'Image Field')
#         # print(Dfx)
#         # Dfx = []
#         # Dfy = []
#         # Imx = []
#         # Imy = []
#     fig.show()
    
# # f = z    
# def z1(x, y):
#     return x, y

# # f = z ^ 2
# def z2(x, y):
#     return x ** 2 - y ** 2, 2 * x * y

# # f = exp(z)
# def z3(x, y):
#     return np.exp(x) * np.cos(y), np.exp(x) * np.sin(y)

# leftbound, rightbound, lowerbound, upperbound = -5, 5, -5, 5
# print_Im(z1, leftbound, rightbound, lowerbound, upperbound)
# print_Im(z2, leftbound, rightbound, lowerbound, upperbound)
# print_Im(z3, leftbound, rightbound, lowerbound, upperbound)


# # cup~
# import numpy as np
# def find_ab(x, y, z):
#     m = np.max([x, y])
#     n = np.min([x, y])
#     A = np.array([[m, 1, 0],
#                   [n, 0, 1]])
#     Continue = True
#     up = 0
#     down = 1
#     while Continue:
#         q = A[up, 0] // A[down, 0]
#         A[up, :] = A[up, :] - q * A[down, :]
#         if A[up, 0] == 0 or A[up, 0] == 0:
#             Continue = False
#         up, down = down, up
#     row = np.where(A[:, 0] != 0)[0]
#     x_y = A[row]
#     print(x_y)
#     factor = x_y[0][0]
#     m_factor = x_y[0][1]
#     n_factor = x_y[0][2]
#     p = z // factor
#     r = z % factor
#     if r != 0:
#         # print('No solution')
#         assert r == 0, 'No solution'
#         return None
#     else:
#         a = m_factor * p
#         b = n_factor * p
#         return a, b

# x = 3
# y = 8
# z = 4
# a, b = find_ab(x, y, z)
# print(f'{np.max([x, y])} : {a}')
# print(f'{np.min([x, y])} : {b}')

