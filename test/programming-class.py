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

# r = 25
# board = np.zeros((r, r))
# cor = np.where(board == 0)
# cordinate = list(zip(cor[0], cor[1]))
# row = np.arange(r)
# # print(row)
# # print(cordinate)
# board_data = pd.DataFrame(index = row, columns = row, data = board)
# print(board_data)

###################################
r = 2
board = np.zeros((r, r))
cor = np.where(board == 0)
cordinate = list(zip(cor[0], cor[1]))
board_brand = np.zeros(r ** 2, dtype = np.uint32)
# board_brand = [0] * (r ** 2)
board_type = [0] * (r ** 2)
board_HP = [0] * (r ** 2)
board_creators = [0] * (r ** 2)

board_data = pd.DataFrame(columns = cordinate, index = ['brand', 'id', 'HP', 'creators'], 
                          data = [board_brand, board_type, board_HP, board_creators])
print(board_data.loc['id'])
print(board_data.loc[['id', 'HP']])
print(board_data.at['id', (1, 0)])
board_data.at['id', (1, 0)] = 1
print(board_data)
def block_manage():
    block_id = np.arange(6)
    block_name = ['air', 'water', 'dirt', 'wood', 'stone', 'steel']
    block_maxHP = [0, 0, 10, 20, 30, 40]
    block_capacity = [5, 2, 3, 5, 6, 7]
    block_DF = [0, 0, 0, 1, 3, 5]
    block_data = pd.DataFrame(columns = block_id, index = ['name', 'max_HP', 'capacity', 'DF'],
                               data = [block_name, block_maxHP,  block_capacity, block_DF])
    print((block_data))
block_manage()