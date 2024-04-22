# game
# Thanks: minecraft, undertale
import numpy as np
import copy
def board_generate(width, length):
    board = np.zeros((width, length), dtype = np.uint32)
    board[0, length // 2] = 1
    board[width - 1, length // 2] = 2
    return board
def print_board(board, player, pointer):
    for row in range(np.size(board, 0)):
        if player == 1 and row == pointer[0]:
            print('\n', ' ' * (pointer[1] * 2), chr(9661), sep = '')
            for column in board[row]:
                if column == 0:
                    print(chr(9723), sep = '', end = ' ' * 1)
                elif column == 1:
                    print(chr(1093), sep = '', end = ' ' * 1)
                elif column == 2:
                    print(chr(916), sep = '', end = ' ' * 1)
                elif column == 3:
                    print(chr(1044), sep = '', end = ' ' * 1)
                else:
                    print(chr(8776), sep = '', end = ' ' * 1)

        elif player == 2 and row == pointer[0]:
            print('')
            for column in board[row]:
                if column == 0:
                    print(chr(9723), sep = '', end = ' ' * 1)
                elif column == 1:
                    print(chr(1093), sep = '', end = ' ' * 1)
                elif column == 2:
                    print(chr(916), sep = '', end = ' ' * 1)
                elif column == 3:
                    print(chr(1044), sep = '', end = ' ' * 1)
                else:
                    print(chr(8776), sep = '', end = ' ' * 1)
            print('\n', ' ' * (pointer[1] * 2), chr(9651), sep = '', end = '')
        else:
            print('')
            for column in board[row]:
                if column == 0:
                    print(chr(9723), sep = '', end = ' ' * 1)
                elif column == 1:
                    print(chr(1093), sep = '', end = ' ' * 1)
                elif column == 2:
                    print(chr(916), sep = '', end = ' ' * 1)
                elif column == 3:
                    print(chr(1044), sep = '', end = ' ' * 1)
                else:
                    print(chr(8776), sep = '', end = ' ' * 1)
            
    return None
def obstacles(board, number):
    width = np.size(board, 0)
    length = np.size(board, 1)
    for i in range(number):
        row = np.random.choice(range(1, width // 2))
        column = np.random.choice(range(length))
        board[row, column] = 3
    for i in range(number):
        row = np.random.choice(range(width // 2, width - 1))
        column = np.random.choice(range(length))
        board[row, column] = 3   
    return board
def tarrain(n, r, kind):
    def sea(r):
        seas = np.ones((r, r), dtype = np.uint32)
        seas = seas * 4
        return seas
    def mountain(r):
        mounts = np.zeros((r, r), dtype = np.uint32)
        for row in range(r):
            for column in range(r):
                possibility = np.e ** ((-(row - r // 2) ** 2 - (column - r // 2) ** 2) / r)
                mounts[row, column] = np.random.choice([0, 3], p = [1- possibility, possibility])
        return mounts
    def lake(r):
        lakes = np.zeros((r, r), dtype = np.uint32)
        for row in range(r):
            for column in range(r):
                possibility = np.e ** ((-(row - r // 2) ** 2 - (column - r // 2) ** 2) / r)
                lakes[row, column] = np.random.choice([0, 4], p = [1- possibility, possibility])
        return lakes
    def rockylake(r):
        rockylakes = np.zeros((r, r), dtype = np.uint32)
        for row in range(r):
            for column in range(r):
                possibility = np.e ** ((-(row - r // 2) ** 2 - (column - r // 2) ** 2) / r)
                rockylakes[row, column] = np.random.choice([0, 3, 4], p = [1- possibility, possibility / 2, possibility / 2])
        return rockylakes
    def island(r):
        islands = np.ones((r, r), dtype = np.uint32)
        islands = islands * 4
        for row in range(r):
            for column in range(r):
                possibility = np.e ** ((-(row - r // 2) ** 2 - (column - r // 2) ** 2) / np.sqrt(r))
                islands[row, column] = np.random.choice([4, 0], p = [1- possibility, possibility])
        return islands
    def shore(r, n = 'left'):
        shores = np.zeros((r, r), dtype = np.uint32)
        column = np.random.choice([r // 2 - 1, r // 2, r // 2 + 1], p = [1/5, 3/5, 1/5])
        for row in range(r):
            t = np.e ** ((-(column - r / 2) ** 2) / r)
            if column == 0:
                left = 0
                middle = np.e ** ((-(column - r / 2) ** 2) / r)
                right = np.e ** ((-(column + 1 - r / 2) ** 2) / r)
            elif column == r - 1:
                left = np.e ** ((-(column - 1 - r / 2) ** 2) / r)
                middle = np.e ** ((-(column - r / 2) ** 2) / r)
                right = 0
            else:
                left = np.e ** ((-(column - 1 - r / 2) ** 2) / r)
                middle = np.e ** ((-(column - r / 2) ** 2) / r)
                right = np.e ** ((-(column + 1 - r / 2) ** 2) / r)
            total = left + middle + right
            move = np.random.choice([-1, 0, 1], p = [left / total, middle / total, right / total])
            column += move
            if n == 'left':
                for i in range(column + 1):
                    shores[row, i] = 4
            else:
                for i in range(column, r):
                    shores[row, i] = 4
        return shores
    def plain(r):
        plains = np.zeros((r, r), dtype = np.uint32)
        return plains
    def hill(r):
        hills = np.zeros((r, r), dtype = np.uint32)
        for i in range(r):
            row = np.random.choice(range(r))
            column = np.random.choice(range(r))
            hills[row, column] = 3
        return hills
    def river(r):
        rivers = np.zeros((r, r), dtype = np.uint32)
        row = np.random.choice([r // 2 - 1, r // 2, r // 2 + 1], p = [1/5, 3/5, 1/5])
        for column in range(r):
            t = np.e ** ((-(row - r / 2) ** 2) / r)
            if row == 1:
                left = 0
                middle = np.e ** ((-(row - r / 2) ** 2) / r)
                right = np.e ** ((-(row + 1 - r / 2) ** 2) / r)
            elif row == r - 2:
                left = np.e ** ((-(row - 1 - r / 2) ** 2) / r)
                middle = np.e ** ((-(row - r / 2) ** 2) / r)
                right = 0
            else:
                left = np.e ** ((-(row - 1 - r / 2) ** 2) / r)
                middle = np.e ** ((-(row - r / 2) ** 2) / r)
                right = np.e ** ((-(row + 1 - r / 2) ** 2) / r)
            total = left + middle + right
            move = np.random.choice([-1, 0, 1], p = [left / total, middle / total, right / total])
            row += move
            rivers[row, column] = 4
            rivers[row - 1, column] = np.random.choice([0, 4], p = [0.3, 0.7])
            rivers[row + 1, column] = np.random.choice([0, 4], p = [0.3, 0.7])
        bridge = np.random.choice(range(1, r // (r // 2)))
        for i in range(bridge):
            column = np.random.choice(range(r))
            rivers[:, column] = 0
        return rivers
    if kind == 'shores and hills':
        for row in range(n):
            for column in range(n):
                name = np.random.choice([mountain, plain, hill, river])
                t = name(r)
                if column == 0:
                    m = copy.deepcopy(t)
                else:
                    m = np.hstack((m, t))
            if row == 0:
                h = copy.deepcopy(m)
            else:
                h = np.vstack((h, m))
        for row in range(n):
            h[row * r : (row + 1) * r, 0 : r] = shore(r)
            h[row * r : (row + 1) * r, (n - 1) * r: n * r] = shore(r, 'right')
        h[0: r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[(n - 1) * r: n * r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[0, (n * r) // 2], h[(n * r - 1), (n * r) // 2] = 1, 2
        # print(h)
        return h
    elif kind == 'mountains':
        for row in range(n):
            for column in range(n):
                name = np.random.choice([mountain, hill])
                t = name(r)
                if column == 0:
                    m = copy.deepcopy(t)
                else:
                    m = np.hstack((m, t))
            if row == 0:
                h = copy.deepcopy(m)
            else:
                h = np.vstack((h, m))
        h[0: r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[(n - 1) * r: n * r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[0, (n * r) // 2], h[(n * r - 1), (n * r) // 2] = 1, 2
        return h
    elif kind == 'marsh':
        for row in range(n):
            for column in range(n):
                name = np.random.choice([river, lake, rockylake], p = [0.2, 0.3, 0.5])
                t = name(r)
                if column == 0:
                    m = copy.deepcopy(t)
                else:
                    m = np.hstack((m, t))
            if row == 0:
                h = copy.deepcopy(m)
            else:
                h = np.vstack((h, m))
        h[0: r, (n // 2) * r : (n // 2 + 1) * r] = island(r)
        h[(n - 1) * r: n * r, (n // 2) * r : (n // 2 + 1) * r] = island(r)
        h[0, (n * r) // 2], h[(n * r - 1), (n * r) // 2] = 1, 2
        return h
    elif kind == 'miniworld':
        for row in range(n):
            for column in range(n):
                name = np.random.choice([mountain, hill, lake, rockylake, river, plain])
                t = name(r)
                if column == 0:
                    m = copy.deepcopy(t)
                else:
                    m = np.hstack((m, t))
            if row == 0:
                h = copy.deepcopy(m)
            else:
                h = np.vstack((h, m))
        h[0: r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[(n - 1) * r: n * r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[0, (n * r) // 2], h[(n * r - 1), (n * r) // 2] = 1, 2
        return h
    elif kind == 'navigation':
        for row in range(n):
            for column in range(n):
                name = np.random.choice([sea, island, rockylake], p = [0.4, 0.5, 0.1])
                t = name(r)
                if column == 0:
                    m = copy.deepcopy(t)
                else:
                    m = np.hstack((m, t))
            if row == 0:
                h = copy.deepcopy(m)
            else:
                h = np.vstack((h, m))
        h[0: r, (n // 2) * r : (n // 2 + 1) * r] = island(r)
        h[(n - 1) * r: n * r, (n // 2) * r : (n // 2 + 1) * r] = island(r)
        h[0, (n * r) // 2], h[(n * r - 1), (n * r) // 2] = 1, 2
        return h
    elif kind == 'chaos':
        for row in range(n):
            for column in range(n):
                name = np.random.choice([mountain, hill, lake, rockylake, river, plain, sea, island])
                t = name(r)
                if column == 0:
                    m = copy.deepcopy(t)
                else:
                    m = np.hstack((m, t))
            if row == 0:
                h = copy.deepcopy(m)
            else:
                h = np.vstack((h, m))
        h[0: r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[(n - 1) * r: n * r, (n // 2) * r : (n // 2 + 1) * r] = plain(r)
        h[0, (n * r) // 2], h[(n * r - 1), (n * r) // 2] = 1, 2
        return h
    else:
        for row in range(n):
            for column in range(n):
                name = plain
                t = name(r)
                if column == 0:
                    m = copy.deepcopy(t)
                else:
                    m = np.hstack((m, t))
            if row == 0:
                h = copy.deepcopy(m)
            else:
                h = np.vstack((h, m))
        h[0, (n * r) // 2], h[(n * r - 1), (n * r) // 2] = 1, 2
        return h
# for i in range(1000):
#     print(chr(i + 2000), end = ' ')
# print(ord('≈'))
# ╳ ~ 9587
# Δ ~ 916 х ~ 1093 ◻ ~ 9723 ▽ ~ 9961 △ ~ 9651 Д ~ 1044 ≈ ~ 8776

def available(board, player):
    # player is 1 or 2
    place = np.where(board == player)
    place_extend = np.where(board == 4)
    place = list(zip(place[0], place[1]))
    place_extend = list(zip(place_extend[0], place_extend[1]))
    place += place_extend
    return place
def nullplace(board):
    place = np.where(board == 0)
    place = list(zip(place[0], place[1]))
    return place
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
        if (pointer[0] - 1, pointer[1]) in available_place:
            av_move.append('w')
        if (pointer[0] + 1, pointer[1]) in available_place:
            av_move.append('s')
        if (pointer[0], pointer[1] - 1) in available_place:
            av_move.append('a')
        if (pointer[0], pointer[1] + 1) in available_place:
            av_move.append('d')
        action = input(f'\n choose actions! {av_move}, \n')
        action = action.strip().lower()
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
        else:
            print('not available place')
def paint_block(board, player, pointer):
    null_space = nullplace(board)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (pointer[0] + i, pointer[1] + j) in null_space:
                board[pointer[0] + i, pointer[1] + j] = player
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
            print('\n Game over, everyone is the winner!')
        elif player1 > player2:
            print('\n Game over, player1 is the winner!')
        else:
            print('\n Game over, player2 is the winner!')
        return True
    else:
        return False
def let_over():
    a = input('End the game? T / press enter to keep on')
    a = a.strip().lower()
    if a == 't':
        return True
    else:
        return False
def start():
    k = input('which kind of map do you want? \n 1:shores and hills; 2:mountains; \n 3:marsh(recommended); 4:miniworld; \n 5:navigation; 6:chaos; \n else:superplain, \n choose a number to enter').strip()
    if k == '1':
        kind = 'shores and hills'
    elif k == '2':
        kind = 'mountains'
    elif k == '3':
        kind = 'marsh'
    elif k == '4':
        kind = 'miniworld'
    elif k == '5':
        kind = 'navigation'
    elif k == '6':
        kind = 'chaos'
    else:
        kind = 'Ciallo~(∠·ω<)⌒★'
    board = tarrain(5, 5, kind)
    # board = obstacles(board, 15)
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
        if let_over():
            player1 = score(board, 1)
            player2 = score(board, 2)
            if player1 == player2:
                print('Game over, everyone is the winner!')
            elif player1 > player2:
                print('Game over, player1 is the winner!')
            else:
                print('Game over, player2 is the winner!')
            return None
start()