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

# def action_move(board, source, player):
    
#     forward = input('请选择你的方向:w/a/s/d')
    
#     '''location: (10, 14)'''
# def av_action_move(board, source, player):
#     row_bound = np.size(board, 0)
#     column_bound = np.size(board, 1)
#     location = player[location]
#     up = np.maximum(location[0], 0)
#     down = np.minimum(location[0], row_bound - 1)
#     left = np.maximum(location[1], 0)
#     right = np.minimum(location[1], column_bound - 1)
#     boardplace_rock = np.where(board[up: down, left: right] == 2)
#     boardplace_water = np.where(board[up: down, left: right] == 1)
#     boardplace_air = np.where(board[up: down, left: right] == 0)
#     source_empty = np.where(source[up: down, left: right] <= 1)
#     source_bridge = np.where(source[up: down, left: right] == 10) #bridge ID
#     air_place = 1

def f(x, y):
    return x + np.cos(- y ** 2)
def solve_ode(f, X, y0):
    x = 0
    y = y0
    Y = np.ones(len(X)) * y0
    for i in range(len(X) - 1):
        Y[i + 1] = Y[i] + (X[i + 1] - X[i]) * f(X[i], Y[i])
    return Y

fig = go.Figure()
for multi in range(5):
    delta = pow(10, -multi)
    X = np.arange(0, 5, delta)
    y0 = 0
    solution = solve_ode(f, X, y0)
    
    fig.add_scatter(x = X, y = solution, mode = 'lines+markers', name = f'delta:{delta}')
fig.show()

def RK_method(f, X, y0, h):
    Y = np.ones(len(X)) * y0
    for i in range(len(X) - 1):
        k1 = f(X[i], Y[i])
        k2 = f(X[i] + h / 2, Y[i] + (h * k1) / 2)
        k3 = f(X[i] + h / 2, Y[i] + (h * k2) / 2)
        k4 = f(X[i] + h, Y[i] + h * k3)
        Y[i + 1] = Y[i] + (h * (k1 + 2 * k2 + 2 * k3 + k4)) / 6
    return Y


def gradient(f, partial, x0, treshold, maxiter):
    x_curr = x0
    X = np.array([x0])
    count = 0
    while True:
        dx = partial(x0)
        x_next = x_curr + dx
        X = np.vstack(X, x_next)
        count += 1

        df = f(x_next) - f(x_curr)
        
        if df < treshold:
            return X.T
        elif count == maxiter:
            return X.T
        
