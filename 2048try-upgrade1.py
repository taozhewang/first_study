import numpy as np
import random
import copy

new_board = np.zeros((4, 4), dtype = np.uint32)
def create_board():
    board = np.zeros((4, 4), dtype = np.uint32 )
    return board

def get_available_space(board):
    available_cell = list(np.where(board == 0))
    available_space = list(zip(available_cell[0], available_cell[1]))
    return available_space

def feed_number(board, available_space):
    if available_space == []:
        return board
    i, j= random.choice(available_space)
    random_number = np.random.choice([2,4], p = [6/7, 1/7])
    board[i, j] = random_number
    return board

def digits(number):
    digit = 0
    while number > 0:
        number = number // 10
        digit += 1
    return digit

def print_board(board):
    print('-' * 29)
    for i in board:
        print('', sep = '', end = ' ')
        for j in i:
            digit = digits(j)
            if digit <= 1:
                print(' ' * 3, j, ' ' * 3, sep = '', end = '')
            elif digit == 2:
                print(' ' * 2, j, ' ' * 3, sep = '', end = '')
            elif digit == 3:
                print(' ' * 2, j, ' ' * 2, sep = '', end = '')
            else:
                print(' ', j, ' ' * 2, sep = '', end = '')
        print('')
        print('-' * 29)

def get_action():
    while True:
        forward = input('请选择一个方向,w/a/s/d.\n').strip().lower()
        if forward not in ['w', 'a', 's', 'd']:
            print('请输入正确的方向!', '\n')
        else:
            return forward
        
def w_indent_zero(board, row, column):
        non_zero = []
        for row in range(4):
            if board[row, column] != 0:
                non_zero.append(board[row, column])
        for row in range(len(non_zero)):
            board[row, column] = non_zero[row]
        for left_row in range(len(non_zero), 4):
            board[left_row, column] = 0
        return board

def move(board, forward, total_score):
    row, column = 0, 0
    if forward == 'w':
        for column in range(4):
            board = w_indent_zero(board, row, column)
            for row in range(3):
                if board[row, column] == board[row + 1, column]:
                    board[row, column] = board[row, column] * 2
                    board[row + 1, column] = 0
                    total_score += board[row, column]
            board = w_indent_zero(board, row, column)
        return board, total_score
    if forward == 'd':
        board, total_score = move(np.rot90(board), 'w', total_score)
        return np.rot90(board, k = 3), total_score
    if forward == 's': 
        board, total_score = move(np.rot90(board, k = 2), 'w', total_score)
        return np.rot90(board, k = 2), total_score
    if forward == 'a':
        board, total_score = move(np.rot90(board, k = 3), 'w', total_score)
        return np.rot90(board), total_score

def correct_move(board, total_score):
    while True:
        new_board = copy.deepcopy(board)
        board, total_score = move(board, get_action(), total_score)
        if not np.array_equal(board, new_board):
            return board, total_score

def horizonal_check(board):
    for i in range(4):
        for j in range(3):
            if board[i, j] == board[i, j+1]:
                return True
    return False

def check_end(board):
    if 0 in board or horizonal_check(board) or horizonal_check(np.rot90(board)):
        return True
    return False

def start_game():
    total_score = 0
    board = create_board()
    print('Welcome to 2048!')
    for _ in range(2):
        board = feed_number(board, get_available_space(board))
    print_board(board)

    while True:
        board, total_score = correct_move(board, total_score)
        board = feed_number(board, get_available_space(board))
        print_board(board)
        print('Scores:', total_score) 
        if not check_end(board):            
            return False
        
start_game()
# something goes wrong again.
# def auto_start(board, in_total_score, depth = 10):
#     n_board = copy.deepcopy(board)
#     examples = {}
#     for indexx in range(5):
#         total_score = in_total_score
#         n_board, total_score, forward = auto_move(n_board, 0, total_score)
#         for _ in range(depth):           
#             n_board, total_score = auto_move(n_board, 1, total_score)
#             n_board = feed_number(n_board, get_available_space(n_board)) 
#             if not check_end(n_board):            
#                 total_score = -1
#                 break
#         examples[indexx] = (total_score, forward) 
#     contemperory = -1
#     for score, forward in examples.values():
#         if score >= contemperory:
#             contemperory = score
#     for score, forward in examples.values():
#         if score == contemperory:
#             print('good way:', forward)
#             break
#     return None

# def auto_move(n_board, p, total_score):
#     while True:
#         new_board = copy.deepcopy(n_board)
#         forward = random.choice(['w', 'a', 's', 'd'])
#         n_board, total_score = move(n_board, forward, total_score)
#         if not np.array_equal(n_board, new_board):
#             if p:
#                 return n_board, total_score
#             return n_board, total_score, forward

# def start_game_with_help():
#     total_score = 0
#     board = create_board()
#     print('Welcome to 2048!')
#     for _ in range(2):
#         board = feed_number(board, get_available_space(board))
#     print_board(board)

#     while True:
#         board, total_score = correct_move(board, total_score)
#         board = feed_number(board, get_available_space(board))
#         print_board(board)
#         print('Scores:', total_score) 
#         if not check_end(board):  
#             print('Game Over!')          
#             return False
#         auto_start(board, total_score, 10)

# start_game_with_help()