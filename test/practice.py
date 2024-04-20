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

# game
import numpy as np
a = np.array([[0, 0, 1, 1, 1, 0, 1], 
              [1, 2, 1, 1, 2, 2, 2],
              [2, 2, 2, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1],
              [2, 0, 2, 1, 1, 2, 2],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0]])

def board_generate(width, length):
    board = np.zeros((width, length), dtype = np.uint32)
    board[0, length // 2] = 1
    board[width - 1, length // 2] = 2
    return board
def print_board(board, player, pointer):
    for row in range(np.size(board, 0)):
        if player == 1 and row == pointer[0]:
            print(' ' * (pointer[1] * 5), chr(9661), sep = '')
            for column in board[row]:
                if column == 0:
                    print(chr(9723), sep = '', end = ' ' * 4)
                elif column == 1:
                    print(chr(1093), sep = '', end = ' ' * 4)
                elif column == 2:
                    print(chr(916), sep = '', end = ' ' * 4)
            print('\n')
        elif player == 2 and row == pointer[0]:
            for column in board[row]:
                if column == 0:
                    print(chr(9723), sep = '', end = ' ' * 4)
                elif column == 1:
                    print(chr(1093), sep = '', end = ' ' * 4)
                elif column == 2:
                    print(chr(916), sep = '', end = ' ' * 4)
            print('\n', ' ' * (pointer[1] * 5), chr(9651), sep = '')
        else:
            for column in board[row]:
                if column == 0:
                    print(chr(9723), sep = '', end = ' ' * 4)
                elif column == 1:
                    print(chr(1093), sep = '', end = ' ' * 4)
                elif column == 2:
                    print(chr(916), sep = '', end = ' ' * 4)
            print('\n')

    return None

# print_board(a, np.array([2, 3]), 2)
# for i in range(1000):
#     print(chr(i + 2000), end = ' ')
# print(ord('▽'))
# ╳ ~ 9587
# Δ ~ 916 х ~ 1093 ◻ ~ 9723 ▽ ~ 9961 △ ~ 9651


def available(board, player):
    # player is 1 or 2
    place = np.where(board == player)
    place = list(zip(place[0], place[1]))
    return place
def nullplace(board):
    place = np.where(board == 0)
    place = list(zip(place[0], place[1]))
    return place
# print(available(a, 1))
def choose_block(board, player):
    width, length = np.size(board, 0), np.size(board, 1)
    if player == 1:
        pointer = np.array([0, length // 2])
    elif player == 2:
        pointer = np.array([width - 1, length // 2])
    available_place = available(board, player)
    while True:
        print_board(board, player, pointer)
        av_move = ['y']
        # print(pointer)
        if (pointer[0] - 1, pointer[1]) in available_place:
            av_move.append('w')
        if (pointer[0] + 1, pointer[1]) in available_place:
            av_move.append('s')
        if (pointer[0], pointer[1] - 1) in available_place:
            av_move.append('a')
        if (pointer[0], pointer[1] + 1) in available_place:
            av_move.append('d')
        # print(av_move)
        action = input(f'choose actions! {av_move}, \n')
        if action == 'y':
            return pointer
        elif action in av_move:
            if action == 'w':
                pointer = pointer + np.array([-1, 0])
            elif action == 'a':
                pointer = pointer + np.array([0, -1])
            elif action == 's':
                pointer = pointer + np.array([1, 0])
            elif action == 'd':
                pointer = pointer + np.array([0, 1]) 
        elif action not in av_move:
            print('not available place')
        else:
            print('please put in "w" / "a" / "s" / "d" / "y" (without "")')
# choose_block(a, 1)
def paint_block(board, player, pointer):
    null_space = nullplace(board)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (pointer[0] + i, pointer[1] + j) in null_space:
                board[pointer[0] + i, pointer[1] + j] = player
    # print_board(board, player, np.array([100, 100]))
    return board

def score(board, player):
    place = np.where(board == player)
    place = list(zip(place[0], place[1]))
    return len(place)
def game_over(board):
    null_place = nullplace(board)
    if null_place == []:
        player1 = score(board, 1)
        player2 = score(board, 2)
        if player1 == player2:
            print('Game over, everyone is the winner!')
        elif player1 > player2:
            print('Game over, player1 is the winner!')
        else:
            print('Game over, player2 is the winner!')
        return True
    else:
        return False
def start():
    board = board_generate(11, 13)
    player = 1
    while True:
        if game_over(board):
            return None

        pointer = choose_block(board, player)
        board = paint_block(board, player, pointer)
        if player == 1:
            player = 2
        else:
            player = 1
start()
