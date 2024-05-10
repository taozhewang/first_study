import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.options.plotting.backend = 'plotly'

# s = pd.Series(['SH', 'BJ', 'London', np.nan, 'SG'], 
#               index = ['a', 'b', 'c', 'd', 'e'])
# print(s)
# s = pd.Series(data = ['SH', 'BJ', 'London', np.nan, 'SG'], 
#               index = ['a', 'b', 'c', 'd', 'e'], 
#               name = 'My_serie',)
# print(s)

def action_move(board, source, player):
    
    forward = input('请选择你的方向:w/a/s/d')
    
    '''location: (10, 14)'''
def av_action_move(board, source, player):
    row_bound = np.size(board, 0)
    column_bound = np.size(board, 1)
    location = player[location]
    up = np.maximum(location[0], 0)
    down = np.minimum(location[0], row_bound - 1)
    left = np.maximum(location[1], 0)
    right = np.minimum(location[1], column_bound - 1)
    boardplace_rock = np.where(board[up: down, left: right] == 2)
    boardplace_water = np.where(board[up: down, left: right] == 1)
    boardplace_air = np.where(board[up: down, left: right] == 0)
    source_empty = np.where(source[up: down, left: right] <= 1)
    source_bridge = np.where(source[up: down, left: right] == 10) #bridge ID
    air_place = 1