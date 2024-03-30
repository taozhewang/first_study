import numpy as np
import random
import copy

board = []
total_score = [0]
def create_board():
    for _ in range(4):
        board.append([0, 0, 0, 0])

def get_available_space():
    #Get available space from the board
    available_space = []
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                available_space.append([i, j])
    return available_space

def feed_number(available_space):
    #feed a number into one available space
    i,j= random.choice(available_space)
    random_number = random.choice([2,2,2,2,2,4])
    board[i][j] = random_number
    # # get_random_number = np.random.randint(0, 1)
    # # number_available = [2, 4]
    # # random_number = number_available[get_random_number]
    # # available_len = len(available_space)
    # # random_space = np.random.randint(0, available_len)
    # y = available_space[random_space][0]
    # x = available_space[random_space][1]
    # board[y][x] = random_number

def digits(number):
    digit = 0
    while number > 0:
        number = number // 10
        digit += 1
    return digit

def print_board():
    print('-' * 29)
    for i in board:
        print('', sep = '', end = '|')
        for j in i:
            digit = digits(j)
            if digit <= 1:
                print(' ' * 3, j, ' ' * 2,sep = '', end = '|')
            elif digit == 2:
                print(' ' * 2, j, ' ' * 2,sep = '', end = '|')
            elif digit == 3:
                print(' ', j, ' ' * 2, sep = '', end = '|')
            else:
                print(' ', j, ' ', sep = '', end = '|')
        print('')
        print('-' * 29)

def get_action():
    while True:
        forward = input('请选择一个方向,w/a/s/d.\n').strip().lower()
        if forward not in ['w', 'a', 's', 'd']:
            print('请输入正确的方向!', '\n')
        else:
            return forward
        
def indent_zero(n, row, column):
    if n == 'w':
        non_zero = []
        for row in range(4):
            if board[row][column] != 0:
                non_zero.append(board[row][column])
        for row in range(len(non_zero)):
            board[row][column] = non_zero[row]
        for left_row in range(len(non_zero),4):
            board[left_row][column] = 0
    if n == 'a':
        non_zero = []
        for column in range(4):
            if board[row][column] != 0:
                non_zero.append(board[row][column])
        for column in range(len(non_zero)):
            board[row][column] = non_zero[column]
        for left_column in range(len(non_zero),4):
            board[row][left_column] = 0
    if n == 's':
        non_zero = []
        for row in range(3, -1, -1):
            if board[row][column] != 0:
                non_zero.append(board[row][column])
        for row in range(3, 3-len(non_zero), -1):
            board[row][column] = non_zero[3-row]
        for left_row in range(3-len(non_zero), -1, -1):
            board[left_row][column] = 0
    if n == 'd':
        non_zero = []
        for column in range(3,-1,-1):
            if board[row][column] != 0:
                non_zero.append(board[row][column])
        for column in range(3, 3-len(non_zero), -1):
            board[row][column] = non_zero[3-column]
        for left_column in range(3-len(non_zero), -1, -1):
            board[row][left_column] = 0

def move(forward):
    row, column = 0, 0
        # f = 0
        # duplicate = []
        # for _ in range(4):
        #     duplicate.append([0, 0, 0, 0])
        # for i in range(4):
        #     for j in range(4):
        #         duplicate[i][j] = board[i][j]
    duplicate = copy.deepcopy(board)
    if forward == 'w':
#let 0 accumulate at one side
        for column in range(4):
            #     non_zero = []
            #     for row in range(4):
            #         if board[row][column] != 0:
            #             non_zero.append(board[row][column])
            #     for row in range(len(non_zero)):
            #         board[row][column] = non_zero[row]
            #     for left_row in range(len(non_zero),4):
            #         board[left_row][column] 
            indent_zero('w', row, column)
            for row in range(3):
                if board[row][column] == board[row+1][column]:
                    board[row][column] = board[row][column] * 2
                    board[row+1][column] = 0
                    total_score[0] += board[row][column]
            indent_zero('w', row, column)
                # non_zero = []
                # for row in range(4):
                #     if board[row][column] != 0:
                #         non_zero.append(board[row][column])
                # for row in range(len(non_zero)):
                #     board[row][column] = non_zero[row]
                # for left_row in range(len(non_zero),4):
                #     board[left_row][column] = 0
#add equal number together
                # if board[0][column] == board[1][column]:
                #     if board[2][column] == board[3][column]:
                #         board[0][column] = board[0][column] + board[1][column]
                #         board[1][column] = board[2][column] + board[3][column]
                #         board[2][column] = 0
                #         board[3][column] = 0
                #     else:
                #         board[0][column] = board[0][column] + board[1][column]
                #         board[1][column] = board[2][column]
                #         board[2][column] = board[3][column]
                #         board[3][column] = 0
                # else:
                #     if board[1][column] == board[2][column]:
                #         board[1][column] = board[1][column] + board[2][column]
                #         board[2][column] = board[3][column]
                #         board[3][column] = 0
                #     else:
                #         if board[2][column] == board[3][column]:
                #             board[2][column] = board[2][column] + board[3][column]
                #             board[3][column] = 0
                #         else:
                #             f += 1

    if forward == 'a':
        for row in range(4):
            indent_zero('a', row, column)
            for column in range(3):
                if board[row][column] == board[row][column+1]:
                    board[row][column] = board[row][column] * 2
                    board[row][column+1] = 0
                    total_score[0] += board[row][column]
            indent_zero('a', row, column)
                # non_zero = []
                # for column in range(4):
                #     if board[row][column] != 0:
                #         non_zero.append(board[row][column])
                # for column in range(len(non_zero)):
                #     board[row][column] = non_zero[column]
                # for left_column in range(len(non_zero),4):
                #     board[row][left_column] = 0

                # if board[row][0] == board[row][1]:
                #     if board[row][2] == board[row][3]:
                #         board[row][0] = board[row][0] + board[row][1]
                #         board[row][1] = board[row][2] + board[row][3]
                #         board[row][2] = 0
                #         board[row][3] = 0
                #     else:
                #         board[row][0] = board[row][0] + board[row][1]
                #         board[row][1] = board[row][2]
                #         board[row][2] = board[row][3]
                #         board[row][3] = 0
                # else:
                #     if board[row][1] == board[row][2]:
                #         board[row][1] = board[row][1] + board[row][2]
                #         board[row][2] = board[row][3]
                #         board[row][3] = 0
                #     else:
                #         if board[row][2] == board[row][3]:
                #             board[row][2] = board[row][2] + board[row][3]
                #             board[row][3] = 0
                #         else:
                #             f += 1
                                    
    if forward == 's':
        for column in range(3, -1, -1):
            indent_zero('s', row, column)
            for row in range(3,0,-1):
                if board[row][column] == board[row-1][column]:
                    board[row][column] = board[row][column] * 2
                    board[row-1][column] = 0
                    total_score[0] += board[row][column]
            indent_zero('s', row, column)
                # non_zero = []
                # for row in range(3, -1, -1):
                #     if board[row][column] != 0:
                #         non_zero.append(board[row][column])
                # for row in range(3, 3-len(non_zero), -1):
                #     board[row][column] = non_zero[3-row]
                # for left_row in range(3-len(non_zero), -1, -1):
                #     board[left_row][column] = 0

                # if board[3][column] == board[2][column]:
                #     if board[1][column] == board[0][column]:
                #         board[3][column] = board[3][column] + board[2][column]
                #         board[2][column] = board[1][column] + board[0][column]
                #         board[1][column] = 0
                #         board[0][column] = 0
                #     else:
                #         board[3][column] = board[3][column] + board[2][column]
                #         board[2][column] = board[1][column]
                #         board[1][column] = board[0][column]
                #         board[0][column] = 0
                # else:
                #     if board[2][column] == board[1][column]:
                #         board[2][column] = board[2][column] + board[1][column]
                #         board[1][column] = board[0][column]
                #         board[0][column] = 0
                #     else:
                #         if board[1][column] == board[0][column]:
                #             board[1][column] = board[1][column] + board[0][column]
                #             board[0][column] = 0
                #         else:
                #             f += 1
                                    
    if forward == 'd':
        for row in range(3,-1,-1):
            indent_zero('d', row, column)
            for column in range(3,0,-1):
                if board[row][column] == board[row][column-1]:
                    board[row][column] = board[row][column] * 2
                    board[row][column-1] = 0
                    total_score[0] += board[row][column]
            indent_zero('d', row, column)
                # non_zero = []
                # for column in range(3,-1,-1):
                #     if board[row][column] != 0:
                #         non_zero.append(board[row][column])
                # for column in range(3, 3-len(non_zero), -1):
                #     board[row][column] = non_zero[3-column]
                # for left_column in range(3-len(non_zero), -1, -1):
                #     board[row][left_column] = 0
                
                # if board[row][3] == board[row][2]:
                #     if board[row][1] == board[row][0]:
                #         board[row][3] = board[row][3] + board[row][2]
                #         board[row][2] = board[row][1] + board[row][0]
                #         board[row][1] = 0
                #         board[row][0] = 0
                #     else:
                #         board[row][3] = board[row][3] + board[row][2]
                #         board[row][2] = board[row][1]
                #         board[row][1] = board[row][0]
                #         board[row][0] = 0
                # else:
                #     if board[row][2] == board[row][1]:
                #         board[row][2] = board[row][2] + board[row][1]
                #         board[row][1] = board[row][0]
                #         board[row][0] = 0
                #     else:
                #         if board[row][1] == board[row][0]:
                #             board[row][1] = board[row][1] + board[row][0]
                #             board[row][0] = 0
                #         else:
                #             f += 1
    same = True
    for i in range(4):
        for j in range(4):
            if duplicate[i][j] != board[i][j]:
                same = False
    if same:
        return False
    return True        
        # if f == 4 or same:
        #     return False
        # return True 

def correct_move():
    while True:
        if move(get_action()):
            return None

def check1():
    for i in range(4):
        if 0 in board[i]:
            return True
    return False

def check2():
    for i in range(4):
        for j in range(3):
            if board[i][j] == board[i][j+1]:
                return True
    for j in range(4):
        for i in range(3):
            if board[i][j] == board[i+1][j]:
                return True
    return False

def check_end():
    if not check1() and not check2():
        print('Game Over!')
        return False
    return True

def start_game():
    create_board()
    print('Welcome to 2048!')
    for _ in range(2):
        feed_number(get_available_space())
    print_board()

    while True:
        correct_move()

        feed_number(get_available_space())
        print_board()
        print('Scores:',total_score[0]) 

        if not check_end():            
            return False
        
start_game()