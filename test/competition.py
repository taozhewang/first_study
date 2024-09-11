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
#%%
#tools
def loc_time(curr_time, gap):
    total_num = 224 #孔数目
    velocity = 1 #龙头速度、匀速
    L1 = 2.86 #龙头到第一个龙身的距离
    L2 = 1.65 #龙身之间的距离
    h = 0.275 #孔到窄边的距离
    d = 0.15 #孔到长边的距离
    gamma = gap / (2 * np.pi) #中间量
    ceta0 = 32 * np.pi #起始位置
    medium = np.sqrt(1 + np.power(ceta0, 2)) #中间量
    constant = ceta0 * medium + np.log(ceta0 + medium) #常量
    
    ceta = np.zeros(total_num) #各孔辐角
    ceta[0] = ceta0
    #计算龙头的位置
    def head_loc(ceta, t = curr_time, constant = constant, velocity = velocity, gamma = gamma):
        medium = np.sqrt(1 + np.power(ceta, 2))
        return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
    solve = root(head_loc, ceta[0])
    ceta[0] = solve.x[0]

    def body_loc(ceta2, ceta1 = ceta[0], l = L1, gamma = gamma):
        return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
    solve = root(body_loc, ceta[0])
    ceta[1] = solve.x[0]
    for idx in np.arange(1, total_num - 1):
        def body_loc(ceta2, ceta1 = ceta[idx], l = L2, gamma = gamma):
            return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
        solve = root(body_loc, ceta[idx] + 2)
        ceta[idx + 1] = solve.x[0] 
        assert ceta[idx + 1] > ceta[idx]

    rou = gamma * ceta
    X = rou * np.cos(ceta) #极坐标换算为直角坐标
    Y = rou * np.sin(ceta)
    omega = np.zeros_like(ceta)

    V = np.zeros_like(ceta)
    omega[0] = - velocity / (gamma * np.sqrt(1 + np.power(ceta[0], 2)))
    for idx in np.arange(total_num - 1):
        omega[idx + 1] = (-ceta[idx] * omega[idx] + ceta[idx + 1] * omega[idx] * np.cos(ceta[idx + 1] - ceta[idx]) + ceta[idx] * ceta[idx + 1] * omega[idx] * np.sin(ceta[idx + 1] - ceta[idx]))
        omega[idx + 1]/= (ceta[idx + 1] + ceta[idx] * ceta[idx + 1] * np.sin(ceta[idx + 1] - ceta[idx]) - ceta[idx] * np.cos(ceta[idx + 1] - ceta[idx]))
    V[1:] = - gamma * omega[1:] * np.sqrt(1 + np.power(ceta[1:], 2))
    V[0] = velocity
    return X, Y, V
# X, Y, V = loc_time(142.473838, 0.55)
# fig = go.Figure()
# fig.add_scatter(x = X, y = Y, mode = 'markers')
# fig.show()
# print(X)
# print(Y)
# print(V)
#%%
# total_time = 300
# total_num = 224
# # # total_time = 30
# # # total_num = 30
# # ceta = np.zeros((total_num, total_time + 1))
# # a1 = 32 * np.pi
# # a2 = 11/(40 * np.pi)
# # a_2 = 1 / a2
# # a3 = 1 / np.sqrt(np.power(a2, 2) + 1)

# # C1 = np.sqrt(1 + np.power(a1, 2))
# # C2 = a1 * C1 + np.log(a1 + C1)

# # ceta[0][0] = a1


# # def p_ceta(ceta, gap, start_angle):
# #     return gap * (ceta - start_angle) / (2 * np.pi)

# def question1():
#     gap = 0.55
#     ceta = np.zeros((total_num, total_time + 1))
#     gamma = gap / (2 * np.pi)
#     ceta0 = 32 * np.pi
#     medium = np.sqrt(1 + np.power(ceta0, 2))
#     velocity = 1

#     constant = ceta0 * medium + np.log(ceta0 + medium)
    
#     ceta[0][0] = ceta0
#     for t in np.arange(1, total_time + 1):
#         def head_loc(ceta, t = t, constant = constant, velocity = velocity, gamma = gamma):
#             medium = np.sqrt(1 + np.power(ceta, 2))
#             return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
#         solve = root(head_loc, ceta[0][t - 1])
#         ceta[0][t] = solve.x[0]
#         assert ceta[0][t] < ceta[0][t - 1]
#     for t in np.arange(total_time + 1):

#         ceta1 = ceta[0][t]
#         def body_loc(ceta2, ceta1 = ceta1, l = 2.86, gamma = gamma):
#             return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
#         solve = root(body_loc, ceta1)
#         ceta[1][t] = solve.x[0]

#         for idx in np.arange(1, total_num - 1):
#             def body_loc(ceta2, ceta1 = ceta[idx][t], l = 1.65, gamma = gamma):
#                 return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
#             solve = root(body_loc, ceta[idx][t] + 2)
#             ceta[idx + 1][t] = solve.x[0]
#             assert ceta[idx + 1][t] > ceta[idx][t]

#     rou = gamma * ceta
#     X = rou * np.cos(ceta)
#     Y = rou * np.sin(ceta)

#     omega = np.zeros_like(ceta)
#     V = np.zeros_like(ceta)
#     omega[0] = - velocity / (gamma * np.sqrt(1 + np.power(ceta[0], 2)))
#     for idx in np.arange(total_num - 1):
#         omega[idx + 1] = (-ceta[idx] * omega[idx] + ceta[idx + 1] * omega[idx] * np.cos(ceta[idx + 1] - ceta[idx]) + ceta[idx] * ceta[idx + 1] * omega[idx] * np.sin(ceta[idx + 1] - ceta[idx]))
#         omega[idx + 1]/= (ceta[idx + 1] + ceta[idx] * ceta[idx + 1] * np.sin(ceta[idx + 1] - ceta[idx]) - ceta[idx] * np.cos(ceta[idx + 1] - ceta[idx]))
#     V[1:] = - gamma * omega[1:] * np.sqrt(1 + np.power(ceta[1:], 2))
#     V[0] = np.ones(total_time + 1) * velocity

#     print(ceta[0])
#     return X, Y, V
# X, Y, V = question1()
# print(X)
# print(Y)
# print(V)

# # fig = go.Figure()
# # for i in range(0, 5):
# #     fig.add_scatter(x = X.T[i], y= Y.T[i], mode = 'markers')
# # fig.show()
# # fig = go.Figure()
# # for i in range(50, 55):
# #     fig.add_scatter(x = X.T[i], y= Y.T[i], mode = 'markers')
# # fig.show()
# # fig = go.Figure()
# # for i in range(100, 105):
# #     fig.add_scatter(x = X.T[i], y= Y.T[i], mode = 'markers')
# # fig.show()
# # fig = go.Figure()
# # fig.add_scatter(x = X[0], y= Y[0], mode = 'markers')
# # fig.add_scatter(x = X[1], y= Y[1], mode = 'markers')
# # fig.add_scatter(x = X[2], y= Y[2], mode = 'markers')
# # fig.show()
# # fs = np.sqrt(np.power(X[0] - X[1], 2) + np.power(Y[0] - Y[1], 2))
# # st = np.sqrt(np.power(X[1] - X[2], 2) + np.power(Y[1] - Y[2], 2))
# # print(fs)
# # print(st)
#%%

# def question2():
#     total_num = 224
#     total_time = 300
#     time_gap = 4
#     gap = 0.55
#     ceta = np.zeros((total_num, total_time * time_gap + 1))
#     gamma = gap / (2 * np.pi)
#     ceta0 = 32 * np.pi
#     medium = np.sqrt(1 + np.power(ceta0, 2))
#     velocity = 1

#     constant = ceta0 * medium + np.log(ceta0 + medium)
    
#     ceta[0][0] = ceta0
#     for t in np.arange(1, total_time *time_gap + 1):
#         def head_loc(ceta, t = t / time_gap, constant = constant, velocity = velocity, gamma = gamma):
#             medium = np.sqrt(1 + np.power(ceta, 2))
#             return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
#         solve = root(head_loc, ceta[0][t])
#         ceta[0][t] = solve.x[0]
#         # assert ceta[0][t] < ceta[0][t - 1]
#     for t in np.arange(total_time * time_gap + 1):

#         ceta1 = ceta[0][t]
#         def body_loc(ceta2, ceta1 = ceta1, l = 2.86, gamma = gamma):
#             return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
#         solve = root(body_loc, ceta1)
#         ceta[1][t] = solve.x[0]

#         for idx in np.arange(1, total_num - 1):
#             def body_loc(ceta2, ceta1 = ceta[idx][t], l = 1.65, gamma = gamma):
#                 return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
#             solve = root(body_loc, ceta[idx][t] + 2)
#             ceta[idx + 1][t] = solve.x[0] 
#             assert ceta[idx + 1][t] > ceta[idx][t]

#     rou = gamma * ceta
#     X = rou * np.cos(ceta)
#     Y = rou * np.sin(ceta)

#     omega = np.zeros_like(ceta)
#     V = np.zeros_like(ceta)
#     omega[0] = - velocity / (gamma * np.sqrt(1 + np.power(ceta[0], 2)))
#     for idx in np.arange(total_num - 1):
#         omega[idx + 1] = (-ceta[idx] * omega[idx] + ceta[idx + 1] * omega[idx] * np.cos(ceta[idx + 1] - ceta[idx]) + ceta[idx] * ceta[idx + 1] * omega[idx] * np.sin(ceta[idx + 1] - ceta[idx]))
#         omega[idx + 1]/= (ceta[idx + 1] + ceta[idx] * ceta[idx + 1] * np.sin(ceta[idx + 1] - ceta[idx]) - ceta[idx] * np.cos(ceta[idx + 1] - ceta[idx]))
#     V[1:] = - gamma * omega[1:] * np.sqrt(1 + np.power(ceta[1:], 2))
#     V[0] = np.ones(total_time * time_gap + 1) * velocity

#     # print(ceta[0])
#     return X, Y, V
# X, Y, V = question2()
# print(X)
# print(Y)
# print(V)

#%%
def Q2():
    def if_crash(A1, A2, Ai1, Ai2, L1, L2, h, d):
        x1, y1 = A1[0], A1[1]
        x2, y2 = A2[0], A2[1]
        P = np.array([x1 + (x1 - x2) * h / L1 + (y2 - y1) * d / L1, y1 + (y1 - y2) * h / L1 + (x1 - x2) * d / L1])
        R = P + (A2 - A1) * (L1 + 2 * h) / L1
        A_k2_P_square = np.sum(np.power(Ai2 - P, 2))
        A_k2_Q_square = np.power(np.dot(P - Ai2, Ai1 - Ai2) / L2, 2)
        Q_P_square = A_k2_P_square - A_k2_Q_square
        A_k2_R_square = np.sum(np.power(Ai2 - R, 2))
        A_k2_S_square = np.power(np.dot(R - Ai2, Ai1 - Ai2) / L2, 2)
        S_R_square = A_k2_R_square - A_k2_S_square
        return Q_P_square <= np.power(d, 2) or S_R_square <= np.power(d, 2)


    def question2(start_time, time_gap, computing_time):
        total_num = 224 #孔数目
        gap = 0.55 #螺距
        velocity = 1 #龙头速度、匀速
        L1 = 2.86 #龙头到第一个龙身的距离
        L2 = 1.65 #龙身之间的距离
        h = 0.275 #孔到窄边的距离
        d = 0.15 #孔到长边的距离
        gamma = gap / (2 * np.pi) #中间量
        ceta0 = 32 * np.pi #起始位置
        medium = np.sqrt(1 + np.power(ceta0, 2)) #中间量

        constant = ceta0 * medium + np.log(ceta0 + medium) #常量
        
        end_t = gamma * constant / (velocity * 2) #龙头到达原点时间
        total_time = int(end_t) #时间取整
        
        ceta = np.zeros((total_num, computing_time + 1)) #各孔各时间的辐角
        ceta[0][0] = ceta0
        for t in np.arange(computing_time + 1):
            #计算龙头的位置
            def head_loc(ceta, t = start_time + t / time_gap, constant = constant, velocity = velocity, gamma = gamma):
                medium = np.sqrt(1 + np.power(ceta, 2))
                return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
            solve = root(head_loc, ceta[0][t])
            ceta[0][t] = solve.x[0]

        for t in np.arange(computing_time + 1):
            #计算身体的位置
            ceta1 = ceta[0][t]
            def body_loc(ceta2, ceta1 = ceta1, l = L1, gamma = gamma):
                return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
            solve = root(body_loc, ceta1)
            ceta[1][t] = solve.x[0]
            for idx in np.arange(1, total_num - 1):
                def body_loc(ceta2, ceta1 = ceta[idx][t], l = L2, gamma = gamma):
                    return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
                solve = root(body_loc, ceta[idx][t] + 2)
                ceta[idx + 1][t] = solve.x[0] 
                assert ceta[idx + 1][t] > ceta[idx][t]

        rou = gamma * ceta
        X = rou * np.cos(ceta) #极坐标换算为直角坐标
        Y = rou * np.sin(ceta)

        D0 = np.power(X[4:] - X[0], 2) + np.power(Y[4:] - Y[0], 2)
        D1 = np.power(X[4:] - X[1], 1) + np.power(Y[4:] - Y[1], 1)
        D2 = np.power(X[4:] - X[2], 2) + np.power(Y[4:] - Y[2], 2)
        R = np.power(np.sqrt(np.power(h, 2) + np.power(d, 2)) + d, 2) + np.power(L2, 2) #可能会有擦碰的范围

        for t in np.arange(computing_time + 1):
            E0 = np.where(D0.T[t] < R)[0] + 4
            E1 = np.where(D1.T[t] < R)[0] + 4
            E2 = np.where(D2.T[t] < R)[0] + 4
            for idx in E0:
                if idx == total_num - 1:
                    continue
                if if_crash(np.array([X[0][t], Y[0][t]]), np.array([X[1][t], Y[1][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L1, L2, h, d):
                    print(idx)
                    print(start_time + t / time_gap)
                    print(total_time)
                    print('E0')
                    return start_time + t / time_gap, True
            for idx in E1:
                if idx == total_num - 1:
                    continue
                if if_crash(np.array([X[0][t], Y[0][t]]), np.array([X[1][t], Y[1][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L1, L2, h, d):
                    print(idx)
                    print(start_time + t / time_gap)
                    print(total_time)
                    print('E1')
                    return start_time + t / time_gap, True
                elif if_crash(np.array([X[1][t], Y[1][t]]), np.array([X[2][t], Y[2][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L2, L2, h, d):
                    print(idx)
                    print(start_time + t / time_gap)
                    print(total_time)
                    print('E1')
                    return start_time + t / time_gap, True
            for idx in E2:
                if idx == total_num - 1:
                    continue
                if if_crash(np.array([X[1][t], Y[1][t]]), np.array([X[2][t], Y[2][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L2, L2, h, d):
                    print(idx)
                    print(start_time + t / time_gap)
                    print(total_time)
                    print('E2')
                    return start_time + t / time_gap, True
        print(f'end_time:{start_time + t / time_gap}')
        print('Not Found')
        return start_time + t / time_gap, False
    start_time = 0 
    time_gap = 1
    while time_gap <= 1e+10:
        start_time, crash = question2(start_time, time_gap, 10)
        if crash:
            start_time -= 1 / time_gap
            time_gap *= 10
        print(f'start_time: {start_time}')
    print(f'stop_time:{start_time + 1/ time_gap}')
# Q2()
#%%
def Q3():
    def if_crash(A1, A2, Ai1, Ai2, L1, L2, h, d):
        x1, y1 = A1[0], A1[1]
        x2, y2 = A2[0], A2[1]
        P = np.array([x1 + (x1 - x2) * h / L1 + (y2 - y1) * d / L1, y1 + (y1 - y2) * h / L1 + (x1 - x2) * d / L1])
        R = P + (A2 - A1) * (L1 + 2 * h) / L1
        A_k2_P_square = np.sum(np.power(Ai2 - P, 2))
        A_k2_Q_square = np.power(np.dot(P - Ai2, Ai1 - Ai2) / L2, 2)
        Q_P_square = A_k2_P_square - A_k2_Q_square
        A_k2_R_square = np.sum(np.power(Ai2 - R, 2))
        A_k2_S_square = np.power(np.dot(R - Ai2, Ai1 - Ai2) / L2, 2)
        S_R_square = A_k2_R_square - A_k2_S_square
        return Q_P_square <= np.power(d, 2), S_R_square <= np.power(d, 2)


    def question3(time_gap, gap):
        total_num = 224 #孔数目
        velocity = 1 #龙头速度、匀速
        L1 = 2.86 #龙头到第一个龙身的距离
        L2 = 1.65 #龙身之间的距离
        h = 0.275 #孔到窄边的距离
        d = 0.15 #孔到长边的距离
        gamma = gap / (2 * np.pi) #中间量
        radius = 4.5 #圆周半径
        ceta0 = radius / gamma + 4 * np.pi #起始位置
        medium = np.sqrt(1 + np.power(ceta0, 2)) #中间量

        constant = ceta0 * medium + np.log(ceta0 + medium) #常量
        
        ceta2 = radius / gamma
        medium2 = np.sqrt(1 + np.power(ceta2, 2))
        end_t = gamma * (constant - ceta2 * medium2 - np.log(ceta2 + medium2)) / (velocity * 2)#龙头到达圆周时间

        total_time = end_t 
        computing_time = int(end_t * time_gap)
        
        ceta = np.zeros((total_num, computing_time + 1)) #各孔各时间的辐角
        ceta[0][0] = ceta0
        for t in np.arange(computing_time + 1):
            #计算龙头的位置
            def head_loc(ceta, t = t / time_gap, constant = constant, velocity = velocity, gamma = gamma):
                medium = np.sqrt(1 + np.power(ceta, 2))
                return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
            solve = root(head_loc, ceta[0][t])
            ceta[0][t] = solve.x[0]


        for t in np.arange(computing_time + 1):
            #计算身体的位置
            ceta1 = ceta[0][t]
            def body_loc(ceta2, ceta1 = ceta1, l = L1, gamma = gamma):
                return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
            solve = root(body_loc, ceta1)
            ceta[1][t] = solve.x[0]
            for idx in np.arange(1, total_num - 1):
                def body_loc(ceta2, ceta1 = ceta[idx][t], l = L2, gamma = gamma):
                    return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
                solve = root(body_loc, ceta[idx][t] + 2)
                ceta[idx + 1][t] = solve.x[0] 
                assert ceta[idx + 1][t] > ceta[idx][t]

        rou = gamma * ceta
        X = rou * np.cos(ceta) #极坐标换算为直角坐标
        Y = rou * np.sin(ceta)

        D0 = np.power(X[4:] - X[0], 2) + np.power(Y[4:] - Y[0], 2)
        D1 = np.power(X[4:] - X[1], 1) + np.power(Y[4:] - Y[1], 1)
        D2 = np.power(X[4:] - X[2], 2) + np.power(Y[4:] - Y[2], 2)
        R = np.power(np.sqrt(np.power(h, 2) + np.power(d, 2)) + d, 2) + np.power(L2, 2) #可能会有擦碰的范围

        for t in np.arange(computing_time + 1):
            E0 = np.where(D0.T[t] < R)[0] + 4
            E1 = np.where(D1.T[t] < R)[0] + 4
            E2 = np.where(D2.T[t] < R)[0] + 4
            for idx in E0:
                if idx == total_num - 1:
                    continue
                crash0, crash1 = if_crash(np.array([X[0][t], Y[0][t]]), np.array([X[1][t], Y[1][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L1, L2, h, d)
                if crash0:
                    print(f'crash: head')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E0')
                    return True
                elif crash1:
                    print(f'crash: body1')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E0')
                    return True
            for idx in E1:
                if idx == total_num - 1:
                    continue
                crash1, crash2 = if_crash(np.array([X[0][t], Y[0][t]]), np.array([X[1][t], Y[1][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L1, L2, h, d)
                if crash1:
                    print(f'crash: head')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E1')
                    return True
                elif crash2:
                    print(f'crash: body1')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E1')
                    return True
                crash2, crash3 = if_crash(np.array([X[1][t], Y[1][t]]), np.array([X[2][t], Y[2][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L2, L2, h, d)
                if crash2:
                    print(f'crash: body1')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E1')
                    return True
                elif crash3:
                    print(f'crash: body2')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E1')
                    return True
            for idx in E2:
                if idx == total_num - 1:
                    continue
                crash2, crash3 = if_crash(np.array([X[1][t], Y[1][t]]), np.array([X[2][t], Y[2][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), L2, L2, h, d)
                if crash2:
                    print(f'crash: body1')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E2')
                    return True
                elif crash3:
                    print(f'crash: body2')
                    print(idx)
                    print(t / time_gap)
                    print(total_time)
                    print('E2')
                    return True
        print(f'end_time:{t / time_gap}')
        print('Not Found')
        return False

    time_gap = 1
    gap = 0.3
    gap_step = 0.01
    while gap_step >= 1e-10:
        crash = question3(time_gap, gap)
        if not crash:
            gap -= gap_step * 2
            gap_step /= 10
        else:
            gap += gap_step
        print(f'curr_gap: {gap}')
    print(f'stop_gap:{gap + gap_step * 10}')
# Q3()

#%%
# def if_crash(Loc1, Loc2, Loc_k1, Loc_k2, L1, L2, h, d):
#     x1, y1 = Loc1[0], Loc1[1]
#     x2, y2 = Loc2[0], Loc2[1]
#     P = np.array([x1 + (x1 - x2) * h / L1 + (y2 - y1) * d / L1, y1 + (y1 - y2) * h / L1 + (x1 - x2) * d / L1])
#     R = P + (Loc2 - Loc1) * (L1 + 2 * h) / L1
#     A_k2_P_square = np.sum(np.power(Loc_k2 - P, 2))
#     A_k2_Q_square = np.power(np.dot(P - Loc_k2, Loc_k1 - Loc_k2) / L2, 2)
#     Q_P_square = A_k2_P_square - A_k2_Q_square
#     A_k2_R_square = np.sum(np.power(Loc_k2 - R, 2))
#     A_k2_S_square = np.power(np.dot(R - Loc_k2, Loc_k1 - Loc_k2) / L2, 2)
#     S_R_square = A_k2_R_square - A_k2_S_square
#     return Q_P_square <= np.power(d, 2) or S_R_square <= np.power(d, 2)


# def question2(gap):
#     total_num = 224

#     time_gap = 100
#     velocity = 1
#     radius = 4.5
#     gamma = gap / (2 * np.pi)
#     ceta0 = radius / gamma + 2 * np.pi
#     medium = np.sqrt(1 + np.power(ceta0, 2))


#     constant = ceta0 * medium + np.log(ceta0 + medium)

#     ceta2 = radius / gamma
#     medium2 = np.sqrt(1 + np.power(ceta2, 2))
#     end_t = gamma * (constant - ceta2 * medium2 - np.log(ceta2 + medium2)) / (velocity * 2)
#     total_time = int(end_t) + 1
    
#     ceta = np.zeros((total_num, total_time * time_gap + 1))
#     ceta[0][0] = ceta0
#     for t in np.arange(1, total_time *time_gap + 1):
#         def head_loc(ceta, t = t / time_gap, constant = constant, velocity = velocity, gamma = gamma):
#             medium = np.sqrt(1 + np.power(ceta, 2))
#             return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
#         solve = root(head_loc, ceta[0][t])
#         ceta[0][t] = solve.x[0]
#         # assert ceta[0][t] < ceta[0][t - 1]
#     for t in np.arange(total_time * time_gap + 1):

#         ceta1 = ceta[0][t]
#         def body_loc(ceta2, ceta1 = ceta1, l = 2.86, gamma = gamma):
#             return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
#         solve = root(body_loc, ceta1)
#         ceta[1][t] = solve.x[0]

#         for idx in np.arange(1, total_num - 1):
#             def body_loc(ceta2, ceta1 = ceta[idx][t], l = 1.65, gamma = gamma):
#                 return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
#             solve = root(body_loc, ceta[idx][t] + 2)
#             ceta[idx + 1][t] = solve.x[0] 
#             assert ceta[idx + 1][t] > ceta[idx][t]

#     rou = gamma * ceta
#     X = rou * np.cos(ceta)
#     Y = rou * np.sin(ceta)

#     D0 = np.power(X[1:] - X[0], 2) + np.power(Y[1:] - Y[0], 2)
#     R = np.power(np.sqrt(np.power(0.275, 2) + np.power(0.15, 2)) + 0.15, 2) + np.power(1.65, 2)
#     D1 = np.power(X[4:] - X[2], 2) + np.power(Y[4:] - Y[2], 2)

#     # fig = go.Figure()
#     # fig.add_scatter(x = X[0], y = Y[0], mode = 'markers')
#     # fig.show()

#     for t in np.arange(total_time * time_gap + 1):
#         too_close = np.where(D0.T[t] < R)[0] + 1
#         print(too_close)
#         for idx in too_close:
#             if if_crash(np.array([X[0][t], Y[0][t]]), np.array([X[1][t], Y[1][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), 2.86, 1.65, 0.275, 0.15):
#                 print(idx)
#                 print(t/ time_gap)
#                 print(total_time)
#                 print(0, 1)
#                 return idx, t/ time_gap
#             elif if_crash(np.array([X[1][t], Y[1][t]]), np.array([X[2][t], Y[2][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), 1.65, 1.65, 0.275, 0.15):
#                 print(idx)
#                 print(t/ time_gap)
#                 print(total_time)
#                 print(1, 2)
#                 return idx, t/ time_gap
#             else:
#                 continue
#         too_close_too = np.where(D1.T[t] < R)[0] + 4
#         for idx in too_close_too:
#             if if_crash(np.array([X[0][t], Y[0][t]]), np.array([X[1][t], Y[1][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), 2.86, 1.65, 0.275, 0.15):
#                 print(idx)
#                 print(t/ time_gap)
#                 print(total_time)
#                 print('else', 0, 1)
#                 return idx, t/ time_gap
#             elif if_crash(np.array([X[1][t], Y[1][t]]), np.array([X[2][t], Y[2][t]]), np.array([X[idx][t], Y[idx][t]]), np.array([X[idx + 1][t], Y[idx + 1][t]]), 1.65, 1.65, 0.275, 0.15):
#                 print(idx)
#                 print(t/ time_gap)
#                 print(total_time)
#                 print('else', 1, 2)
#                 return idx, t/ time_gap
#             else:
#                 continue
#     print('Not Found')
#     return 0, 0

# # for gap in np.arange(0.4498, 0.45, 0.000001):
# #     question2(gap)
# #     print(f'gap:{gap}')
# gap = 0.449806
# question2(gap)
# print(f'gap:{gap}')
#%%
def question2(gap):
    total_num = 224
    time_gap = 10
    velocity = 1
    radius = 4.5
    gamma = gap / (2 * np.pi)
    ceta0 = radius / gamma + 2 * np.pi
    medium = np.sqrt(1 + np.power(ceta0, 2))
    
    constant = ceta0 * medium + np.log(ceta0 + medium)

    ceta2 = radius / gamma
    medium2 = np.sqrt(1 + np.power(ceta2, 2))
    end_t = gamma * (constant - ceta2 * medium2 - np.log(ceta2 + medium2)) / (velocity * 2)
    total_time = int(end_t) + 4
    
    ceta = np.zeros((total_num, total_time * time_gap + 1))
    ceta[0][0] = ceta0
    for t in np.arange(1, total_time *time_gap + 1):
        def head_loc(ceta, t = t / time_gap, constant = constant, velocity = velocity, gamma = gamma):
            medium = np.sqrt(1 + np.power(ceta, 2))
            return ceta * medium + np.log(ceta + medium) + 2 * t * velocity / gamma - constant
        solve = root(head_loc, ceta[0][t])
        ceta[0][t] = solve.x[0]
        # assert ceta[0][t] < ceta[0][t - 1]
    for t in np.arange(total_time * time_gap + 1):

        ceta1 = ceta[0][t]
        def body_loc(ceta2, ceta1 = ceta1, l = 2.86, gamma = gamma):
            return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
        solve = root(body_loc, ceta1)
        ceta[1][t] = solve.x[0]

        for idx in np.arange(1, total_num - 1):
            def body_loc(ceta2, ceta1 = ceta[idx][t], l = 1.65, gamma = gamma):
                return np.power(ceta1, 2) + np.power(ceta2, 2) - 2 * ceta1 * ceta2 * np.cos(ceta2 - ceta1) - np.power(l / gamma, 2)
            solve = root(body_loc, ceta[idx][t] + 2)
            ceta[idx + 1][t] = solve.x[0] 
            assert ceta[idx + 1][t] > ceta[idx][t]

    rou = gamma * ceta
    X = rou * np.cos(ceta)
    Y = rou * np.sin(ceta)

    circle_x = np.arange(-4.5, 4.5, 0.01)
    circle_y = np.sqrt(np.power(4.5, 2) - np.power(circle_x, 2))
    fig = go.Figure()
    fig.add_scatter(x = X[0], y = Y[0], mode = 'lines')
    fig.add_scatter(x = circle_x, y = circle_y, mode = 'lines')
    fig.add_scatter(x = circle_x, y =  - circle_y, mode = 'lines')
    fig.add_scatter(x = -X[0], y = -Y[0], mode = 'lines')
    fig.show()
# question2(1.7)

def radius_ensure(r):
    gap = 1.7
    gamma = gap / (2 * np.pi)
    phi = r / gamma
    k = np.tan(phi + np.arctan(phi))
    ke = np.tan(phi)
    R1 = 2.86
    def limit_point(fi):
        return np.square(gamma * fi * np.cos(fi) - (r / 3) * np.cos(phi)) + np.square(gamma * fi * np.sin(fi) - (r / 3) * np.sin(phi)) - np.square(R1)
    x_A = (r / 3) * (2 * k * ke * np.sin(phi) - (k - 3 * ke) * np.cos(phi)) / (ke - k)
    y_A = r * np.sin(phi) - (2 * r / 3) * (ke * np.sin(phi) + np.cos(phi)) / (ke - k)
    R2 = np.sqrt(np.square(x_A + (r / 3) * np.cos(phi)) + np.square(y_A + (r / 3) * np.sin(phi))) + 2.86
    def smooth(fi):
        return np.square(x_A + gamma * fi * np.cos(fi)) + np.square(y_A + gamma * fi * np.sin(fi)) - np.square(R2)
    solve1 = root(limit_point, phi)
    solve2 = root(smooth, phi)
    return solve1.x - solve2.x
solve = root(radius_ensure, 4.2)
print(solve)
print(solve.x)
