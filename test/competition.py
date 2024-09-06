import numpy as np
import scipy as sc
from scipy.optimize import root
import plotly.graph_objects as go
#%%
# total_time = 300
# total_num = 224
# # total_time = 30
# # total_num = 30
# ceta = np.zeros((total_num, total_time + 1))
# a1 = 32 * np.pi
# a2 = 11/(40 * np.pi)
# a_2 = 1 / a2
# a3 = 1 / np.sqrt(np.power(a2, 2) + 1)

# C1 = np.sqrt(1 + np.power(a1, 2))
# C2 = a1 * C1 + np.log(a1 + C1)

# ceta[0][0] = a1

# for t in np.arange(0, total_time, 1):
#     for idx in np.arange(0, total_num - 1):
#         def loc_func(ceta2):
#             return np.power(ceta[idx][t], 2) + np.power(ceta2, 2) - 2 * ceta[idx][t] * ceta2 * np.cos(ceta2 - ceta[idx][t]) - 1.65 / a_2
#         solution = root(loc_func, ceta[idx][t] + 1)
#         ceta[idx + 1][t] = solution.x[0]
#     def head_loc(ceta):
#         medium = np.sqrt(1 + np.power(ceta, 2))
#         return ceta * medium + np.log(ceta + medium) + 2 * (t + 1) * a_2 - C2
#     solve = root(head_loc, ceta[0][t])
#     ceta[0][t + 1] = solve.x[0]
# for idx in np.arange(0, total_num - 1):    
#     def loc_func(ceta2):
#         return np.power(ceta[idx][total_time], 2) + np.power(ceta2, 2) - 2 * ceta[idx][total_time] * ceta2 * np.cos(ceta2 - ceta[idx][total_time]) - 1 / a_2
#     solution = root(loc_func, ceta[idx][total_time] + 1)
#     ceta[idx + 1][total_time] = solution.x[0]

# print(ceta)

# def x_y(rou, ceta):
#     return (rou * np.cos(ceta), rou * np.sin(ceta))
# fig = go.Figure()
# X, Y = x_y(ceta[0] * a2, ceta[0])
# fig.add_scatter(x = X, y = Y, mode = 'lines+markers')
# fig.show()

#%%
total_time = 300
total_num = 224
# total_time = 30
# total_num = 30
ceta = np.zeros((total_num, total_time + 1))
a1 = 32 * np.pi
a2 = 11/(40 * np.pi)
a_2 = 1 / a2
a3 = 1 / np.sqrt(np.power(a2, 2) + 1)

C1 = np.sqrt(1 + np.power(a1, 2))
C2 = a1 * C1 + np.log(a1 + C1)

ceta[0][0] = a1


def p_ceta(ceta, gap, start_angle):
    return gap * (ceta - start_angle) / (2 * np.pi)

def question1():
    gap = 0.55
    ceta = np.zeros((total_num, total_time + 1))
    gamma = gap / (2 * np.pi)
    ceta0 = 32 * np.pi
    medium = np.sqrt(1 + np.power(ceta0, 2))
    velocity = 1

    constant = ceta0 * medium + np.log(ceta0 + medium)
    
    ceta[0][0] = ceta0
    for t in np.arange(1, total_time + 1):
        def head_loc(ceta, t = t, constant = constant, velocity = velocity, gamma = gamma):
            medium = np.sqrt(1 + np.power(ceta, 2))
            return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
        solve = root(head_loc, ceta[0][t - 1])
        ceta[0][t] = solve.x[0]
        assert ceta[0][t] < ceta[0][t - 1]
    for t in np.arange(total_time + 1):
        l = 2.86
        ceta1 = ceta[0][t]
        def body_loc(ceta2, ceta1 = ceta1, l = l, gamma = gamma):
            return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
        solve = root(body_loc, ceta1)
        ceta[1][t] = solve.x[0]
        l = 1.65
        for idx in np.arange(1, total_num - 1):
            ceta1 = ceta[idx][t]
            def body_loc(ceta2, ceta1 = ceta1, l = l, gamma = gamma):
                return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
            solve = root(body_loc, ceta[idx][t])
            ceta[idx + 1][t] = solve.x[0]
            assert ceta[idx + 1][t] > ceta[idx][t]
    rou = gamma * ceta
    X = rou * np.cos(ceta)
    Y = rou * np.sin(ceta)

    omega = np.zeros_like(ceta)
    V = np.zeros_like(ceta)
    omega[0] = - velocity / (gamma * np.sqrt(1 + np.power(ceta[0], 2)))
    for idx in np.arange(total_num - 1):
        omega[idx + 1] = (-ceta[idx] * omega[idx] + ceta[idx + 1] * omega[idx] * np.cos(ceta[idx + 1] - ceta[idx]) + ceta[idx] * ceta[idx + 1] * omega[idx] * np.sin(ceta[idx + 1] - ceta[idx]))
        omega[idx + 1]/= (ceta[idx + 1] + ceta[idx] * ceta[idx + 1] * np.sin(ceta[idx + 1] - ceta[idx]) - ceta[idx] * np.cos(ceta[idx + 1] - ceta[idx]))
    V[1:] = - gamma * omega[1:] * np.sqrt(1 + np.power(ceta[1:], 2))
    V[0] = np.ones(total_time + 1) * velocity

    print(ceta[0])
    return X, Y, V
X, Y, V = question1()
print(X)
print(Y)
print(V)

fig = go.Figure()
for i in range(0, 5):
    fig.add_scatter(x = X.T[i], y= Y.T[i], mode = 'markers')
fig.show()
fig = go.Figure()
for i in range(50, 55):
    fig.add_scatter(x = X.T[i], y= Y.T[i], mode = 'markers')
fig.show()
fig = go.Figure()
for i in range(100, 105):
    fig.add_scatter(x = X.T[i], y= Y.T[i], mode = 'markers')
fig.show()
fig = go.Figure()
fig.add_scatter(x = X[0], y= Y[0], mode = 'markers')
fig.add_scatter(x = X[1], y= Y[1], mode = 'markers')
fig.add_scatter(x = X[2], y= Y[2], mode = 'markers')
fig.show()


