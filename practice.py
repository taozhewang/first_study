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


# Q2?: create a sudoku game
import numpy as np
import random
import copy
def sudoku_building(sudoku, x, y):
    available_numbers = []
    for i in range(9):
        n = available(sudoku, i + 1, x, y)
        if n:
            available_numbers.append(n)
    if available_numbers == []:
        return False, sudoku
    number = random.choice(available_numbers)
    sudoku[y][x] = number
    return True, sudoku

def sudoku_create():
    sudoku = np.zeros((9, 9), dtype=np.uint32)
    for y in range(9):
        for x in range(9):
            success, sudoku = sudoku_building(sudoku, x, y)
            if not success:
                return success, sudoku
    return success, sudoku

def available(sudoku, i, x, y):
    smaller = np.array([[sudoku[yy][xx] for yy in range(3 * (y // 3), 3 * (1 + y // 3))] for xx in range(3 * (x // 3), 3 * (1 + x // 3))])
    if i in sudoku[y] or i in sudoku.T[x] or i in smaller:
        return False
    else:
        return i



def game_forming(sudoku):
    n = 40
    i = 0
    a = [(i, j) for j in range(9) for i in range(9)]
    while i < n:
        x , y = random.choice(a)
        if sudoku[y, x] != 0:
            sudoku[y, x] = 0
            i += 1
    return sudoku
# sudoku = game_forming(sudoku)
# print(sudoku)

def apparatus_game_checking(build_sudoku, ker_space):
    new_sudoku = copy.deepcopy(build_sudoku)
    while True:
        for y, x in zip(ker_space[0], ker_space[1]):
            success, new_sudoku = sudoku_building(new_sudoku, x, y)
            if not success:
                break
    return new_sudoku

def game_checking(build_sudoku):
    ker_space = list(np.where(build_sudoku == 0))
    
    new_sudoku = apparatus_game_checking(build_sudoku, ker_space)
    sudoku_copy = copy.deepcopy(new_sudoku)
    for _ in range(1):
        new_sudoku = apparatus_game_checking(build_sudoku, ker_space)
        if not np.equal(sudoku_copy, new_sudoku):
            return False
    return True

def more_complete_game(sudoku):
    # while True:
        build_sudoku = game_forming(sudoku)
        print(build_sudoku)
        # if game_checking(sudoku):
        #     break
    # return build_sudoku

success, sudoku = sudoku_create()
while not success:
    success, sudoku = sudoku_create()
print(sudoku)
game = more_complete_game(sudoku)
# print(game)