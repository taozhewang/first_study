import numpy as np
import pandas as pd
'''
for board:
0 is for ground
1 is for sea
2 is for mountains

for source:
0 is for air
1 is for wood
2 is for stone
3 is for iron
4 is for water

for being:
0 is for nobody

'''
def plain(r, q = 0):
    plains = np.zeros((r, r))
    return plains

def hill(r, density):
    density = int(density) * 2
    hills = np.zeros((r, r))
    rows = np.random.randint(r, size = density)
    columns = np.random.randint(r, size = density)
    for x, y in zip(rows, columns):
        hills[x, y] = 2
    return hills

def mountain(r, size):
    mountains = np.zeros((r, r))
    for row in range(r):
        for column in range(r):
            distance = (row - r / 2) ** 2 + (column - r / 2) ** 2
            p_ = np.e ** (-distance / (size * 2))
            mountains[row, column] = np.random.choice([0, 2], p = [1 - p_, p_])
    return mountains

def lake(r, size):
    lakes = np.zeros((r, r))
    for row in range(r):
        for column in range(r):
            distance = (row - r / 2) ** 2 + (column - r / 2) ** 2
            k = np.maximum(distance - size ** 2, 0)
            p_ = 1 / ((k * (k + size / 2)) + 1)
            lakes[row, column] = np.random.choice([0, 1], p = [1 - p_, p_])
    lakes[r // 2, r // 2] = 3
    return lakes

# print(plain(11), hill(11, 16), mountain(11, 8), lake(11, 9), sep ='\n')

def river(board):
    rmax = np.size(board, 0)
    cmax = np.size(board, 1)
    place = np.where(board == 3)
    start_place = list(zip(place[0], place[1]))
    f = np.array([[-1, -1, -1,  0, 0, 0,  1, 1, 1], 
                  [-1,  0,  1, -1, 0, 1, -1, 0, 1]])

    # print(start_place)
    for row, column in start_place:
        # print(row, column)
        random_r = np.random.choice([-1, 1])
        random_c = np.random.choice([-1, 1])
        last_forward = np.array([random_r, random_c])
        while row > 0 and row < rmax - 1 and column > 0 and column < cmax - 1:
            board[row, column] = 1
            board[row - 1, column] = np.random.choice([board[row - 1, column], 1], p = [0.4, 0.6])
            board[row + 1, column] = np.random.choice([board[row + 1, column], 1], p = [0.4, 0.6])
            board[row, column - 1] = np.random.choice([board[row, column - 1], 1], p = [0.4, 0.6])
            board[row, column + 1] = np.random.choice([board[row, column + 1], 1], p = [0.4, 0.6])

            h = np.dot(last_forward, f)
            h = np.maximum(h, np.zeros(9))
            h = h / np.sum(h)
            # print(h)
            forward_key = np.random.choice(range(9), p = h)

            forward = f.T[forward_key, :]
            row += forward[0]
            column += forward[1]
            last_forward = forward
            a = 0
            a += board[np.maximum(row - 1, 0), column] + board[np.minimum(row + 1, rmax - 1), column]
            a += board[row, np.maximum(column - 1, 0)] + board[row, np.minimum(column + 1, cmax - 1)]
            # print(a)
            if a >= 5:
                break
    return board

def tarrain_origin(rdim, cdim, r):
    board = np.zeros((rdim * r, cdim * r))
    for row in range(rdim):
        for column in range(cdim):
            t = np.random.choice([plain, hill, mountain, lake], p = [0.175, 0.325, 0.325, 0.175])
            board[row * r : (row + 1) * r, column * r : (column + 1) * r] = t(r, np.sqrt(r))
    return river(board)

def chest(board, source, being):
    pl = np.where(board == 0)
    place = list(zip(pl[0], pl[1]))
    sc = np.where(source == 0)
    sc_place = list(zip(sc[0], sc[1]))
    be = np.where(being == 0)
    be_place = list(zip(be[0], be[1]))
    av_place = np.intersect1d(place, sc_place, be_place)
    choice_place = np.random.choice(av_place)
    source[choice_place] = 10
    return board

def waters(board, source):
    pl = np.where(board == 1)
    place = list(zip(pl[0], pl[1]))
    for x, y in place:
        source[x, y] = 4
    return source

def stone_iron(board, source):
    pl = np.where(board == 2)
    place = list(zip(pl[0], pl[1]))
    rmax = np.size(board, 0)
    cmax = np.size(board, 1)
    for x, y in place:
        source[x, y] = 2
        if board[np.maximum(x - 1, 0), y] + board[np.minimum(x + 1, rmax - 1), y] + board[x, np.maximum(y - 1, 0)] + board[x, np.minimum(y + 1, cmax - 1)] >= 7:
            source[x, y] = 3
    return source
    
def forest(board, source):
    pl = np.where(board == 0)
    place = list(zip(pl[0], pl[1]))
    rmax = np.size(board, 0)
    cmax = np.size(board, 1)
    for x, y in place:
        if np.sum(board[np.maximum(x - 2, 0) : np.minimum(x + 2, rmax - 1), np.maximum(y - 2, 0) : np.minimum(y + 2, cmax - 1)]) <= 1:
            source[x, y] = 1
    return source

def source_origin(board):
    rmax = np.size(board, 0)
    cmax = np.size(board, 1)
    source = np.zeros((rmax, cmax))
    source = waters(board, source)
    source = stone_iron(board, source)
    source = forest(board, source)
    return source

def player_move(board: np.ndarray, source: np.ndarray, player: int, players_location: dict):
    player_idx, player_idy = players_location[player]
    left_bound = np.max(0, player_idx - 1)
    right_bound = np.min(np.size(board, 0), player_idx + 1)
    up_bound = np.max(0, player_idy - 1)
    low_bound = np.min(np.size(board, 0), player_idy + 1)
    board_movable_places = np.where(board[up_bound : low_bound, left_bound : right_bound] == 0)
    board_movable_places = list(zip(board_movable_places[0], board_movable_places[1]))
    source_movable_places = np.where(source[up_bound : low_bound, left_bound : right_bound] <= 1)
    source_movable_places = list(zip(source_movable_places[0], source_movable_places[1]))
    movable_places = np.intersect1d(board_movable_places, source_movable_places)
    movable_places = '-'.join('x_y' for x, y in movable_places)
    available_chances = 10
    while True:
        print(f'chances: {available_chances}')
        forward = input('choose one forward to move: w/a/s/d').strip().lower()
        if forward == 'w':
            if '0_1' in movable_places:
                players_location[player] = (player_idx - 1, player_idy)
                return players_location
            else:
                available_chances -= 1
                print('not available space')
        elif forward == 's':
            if '2_1' in movable_places:
                players_location[player] = (player_idx + 1, player_idy)
                return players_location
            else:
                available_chances -= 1
                print('not available space') 
        elif forward == 'a':
            if '1_0' in movable_places:
                players_location[player] = (player_idx, player_idy - 1)
                return players_location
            else:
                available_chances -= 1
                print('not available space') 
        elif forward == 'd':
            if '1_2' in movable_places:
                players_location[player] = (player_idx, player_idy + 1)
            else:
                available_chances -= 1
                print('not available space') 
        elif available_chances > 0:
            available_chances -= 1
            print('not proper forward')
        else:
            print('too many fails')
            return players_location
# def player_attack()
board = tarrain_origin(7, 3, 8)
source = source_origin(board)
for i in range(np.size(board, 0)):
    print(board[i])
print()
for i in range(np.size(source, 0)):
    print(source[i])

