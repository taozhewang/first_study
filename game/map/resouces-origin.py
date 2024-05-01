import numpy as np
import pandas as pd

# resouces: forests, stones, iron, chest
# water, wood, stones, irons, tools comes from resources
# nothing: 0; forest_id: 1; stones_id: 2; iron_id: 3; water_id: 4; chest_id: 10
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

