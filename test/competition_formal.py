import numpy as np
import scipy as sc
from scipy.optimize import root
import plotly.graph_objects as go
from scipy.optimize import minimize
#%%
def Q1():
    def question1():
        total_num = 224
        total_time = 300
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

        omega = np.zeros_like(ceta)
        V = np.zeros_like(ceta)
        omega[0] = - velocity / (gamma * np.sqrt(1 + np.power(ceta[0], 2)))
        for idx in np.arange(total_num - 1):
            omega[idx + 1] = (-ceta[idx] * omega[idx] + ceta[idx + 1] * omega[idx] * np.cos(ceta[idx + 1] - ceta[idx]) + ceta[idx] * ceta[idx + 1] * omega[idx] * np.sin(ceta[idx + 1] - ceta[idx]))
            omega[idx + 1]/= (ceta[idx + 1] + ceta[idx] * ceta[idx + 1] * np.sin(ceta[idx + 1] - ceta[idx]) - ceta[idx] * np.cos(ceta[idx + 1] - ceta[idx]))
        V[1:] = - gamma * omega[1:] * np.sqrt(1 + np.power(ceta[1:], 2))
        V[0] = np.ones(total_time + 1) * velocity
        return X, Y, V
    X, Y, V = question1()
    print(X)
    print(Y)
    print(V)

    fig = go.Figure()
    fig.add_scatter(x = X[30], y = Y[30], mode = 'markers+lines')
    fig.show()

    return
Q1()
#%%
# 第二问：if_crash 函数用于判断A1、A2所在板凳是否会与Ai1、Ai2板凳碰撞
# question2函数用于生成从start_time 到 end_t 的一系列点列，并判断是否有碰撞
# 先通过从0s开始，每秒计算一次，得到大致碰撞时间范围，再从该范围内调整时间步长，得到更详细的时间
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
        # gap = 0.450337
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
            start_time -= 2 / time_gap
            time_gap *= 10
        print(f'start_time: {start_time}')
    print(f'stop_time:{start_time + 1/ time_gap}')
# Q2()
#%%
#tools for Q2
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
X, Y, V = loc_time(412.473838, 0.55)
# # #X, Y, V = loc_time(226.78, 0.450337)
fig = go.Figure()
fig.add_scatter(x = X, y = Y, mode = 'markers')
fig.show()
print(X)
print(Y)
print(V)
print(np.sqrt(np.square(X) + np.square(Y)))
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
        D1 = np.power(X[4:] - X[1], 2) + np.power(Y[4:] - Y[1], 2)
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

    time_gap = 10
    gap = 0.4
    gap_step = 0.01
    while gap_step >= 1e-7:
        crash = question3(time_gap, gap)
        if not crash:
            gap -= gap_step * 2
            gap_step /= 10
            # time_gap += 10
        else:
            gap += gap_step
        print(f'curr_gap: {gap}')
    print(f'stop_gap:{gap + gap_step * 10}')
# Q3()
# 0.450337

#


def Q4():
    # def radius_ensure(r):
    #     gap = 1.7
    #     gamma = gap / (2 * np.pi)
    #     phi = r / gamma
    #     k = np.tan(phi + np.arctan(phi))
    #     ke = np.tan(phi)
    #     R1 = 2.86
    #     def limit_point(fi):
    #         return np.square(gamma * fi * np.cos(fi) - (r / 3) * np.cos(phi)) + np.square(gamma * fi * np.sin(fi) - (r / 3) * np.sin(phi)) - np.square(R1)
    #     x_A = (r / 3) * (2 * k * ke * np.sin(phi) - (k - 3 * ke) * np.cos(phi)) / (ke - k)
    #     y_A = r * np.sin(phi) - (2 * r / 3) * (ke * np.sin(phi) + np.cos(phi)) / (ke - k)
    #     R2 = np.sqrt(np.square(x_A + (r / 3) * np.cos(phi)) + np.square(y_A + (r / 3) * np.sin(phi))) + 2.86
    #     def smooth(fi):
    #         return np.square(x_A + gamma * fi * np.cos(fi)) + np.square(y_A + gamma * fi * np.sin(fi)) - np.square(R2)
    #     solve1 = root(limit_point, phi)
    #     solve2 = root(smooth, phi)
    #     return solve1.x[0] - solve2.x[0]
    # solve = root(radius_ensure, 4.27)
    # radius = solve.x[0]
    # r = radius
    # print(r)
    r = 4.29
    # r = 6
    gap = 1.7
    v = 1
    total_num = 224
    total_time = 201
    gamma = gap / (2 * np.pi)
    phi = r / gamma

    k = np.tan(phi + np.arctan(phi))
    ke = np.tan(phi)
    x_A = (r / 3) * (2 * k * ke * np.sin(phi) - (k - 3 * ke) * np.cos(phi)) / (ke - k)
    y_A = r * np.sin(phi) - (2 * r / 3) * (ke * np.sin(phi) + np.cos(phi)) / (ke - k)
    x_C = - (r / 3) * np.cos(phi)
    y_C = - (r / 3) * np.sin(phi)
    r_A = np.sqrt(np.square(x_A - x_C) + np.square(y_A - y_C))
    x_B = (3 * x_C - x_A) / 2
    y_B = (3 * y_C - y_A) / 2
    r_B = r_A / 2
    x_E = r * np.cos(phi)
    y_E = r * np.sin(phi)
    x_F = -x_E
    y_F = -y_E
    
    alpha = np.arccos((2 * r / 3) / r_A)
    t1 = 0
    t2 = r_A * (np.pi - 2 * alpha) / v
    t3 = t2 * 3 / 2
    print(t2)
    print(t3)

    start_time = 100
    end_time = 100
    def head_locs(start_time, end_time, r):
        ceta_head = np.zeros(end_time - start_time + 1)

        ceta_head1 = np.zeros(start_time + 1)
        medium = np.sqrt(1 + np.square(phi))
        constant = phi * medium + np.log(phi + medium)
        
        ceta_head1[-1] = phi
        for t in np.arange(1, start_time + 1):
            def head_loc(fi):
                m = np.sqrt(1 + np.square(fi))
                return fi * m + np.log(fi + m) - constant - 2 * v * t / gamma
            solve = root(head_loc, ceta_head1[-t])

            ceta_head1[-t - 1] = solve.x[0]
        
        rou_head1 = ceta_head1 * gamma
        X_head1 = rou_head1 * np.cos(ceta_head1)
        Y_head1 = rou_head1 * np.sin(ceta_head1)

        ceta_head2 = np.zeros(int(t2))
        rou_head2 = np.zeros(int(t2))
        X_head2 = np.zeros(int(t2))
        Y_head2 = np.zeros(int(t2))
        ceta_head2[0] = phi
        for idt, t in enumerate(np.arange(1, int(t2) + 1)):
            A_to_Q = np.array([x_E - x_A, y_E - y_A]) * r_A * np.cos(v * t / r_A) / r_A
            Q_to_P = np.array([y_E - y_A, x_A - x_E]) * r_A * np.sin(v * t / r_A) / r_A
            x_P, y_P = np.array([x_A, y_A]) + A_to_Q + Q_to_P
            rou_head2[idt] = np.sqrt(np.square(x_P) + np.square(y_P))
            X_head2[idt] = x_P
            Y_head2[idt] = y_P
            E_P_square = np.square(x_P - x_E) + np.square(y_P - y_E)
            ceta_head2[idt] = phi - np.arccos((np.square(rou_head2[idt]) + np.square(r) - E_P_square) / (2 * rou_head2[idt] * r))

        ceta_head3 = np.zeros(int(t3) - int(t2))
        rou_head3 = np.zeros(int(t3) - int(t2))
        X_head3 = np.zeros(int(t3) - int(t2))
        Y_head3 = np.zeros(int(t3) - int(t2))
        for idt, t in enumerate(np.arange(int(t2) + 1, int(t3) + 1)):
            t = t3 - t
            B_to_Q = np.array([x_F - x_B, y_F - y_B]) * r_B * np.cos(v * t / r_B) / r_B
            Q_to_P = np.array([y_F - y_B, x_B - x_F]) * r_B * np.sin(v * t / r_B) / r_B
            x_P, y_P = np.array([x_B, y_B]) + B_to_Q + Q_to_P
            rou_head3[idt] = np.sqrt(np.square(x_P) + np.square(y_P))
            X_head3[idt] = x_P
            Y_head3[idt] = y_P
            F_P_square = np.square(x_P - x_F) + np.square(y_P - y_F)
            ceta_head3[idt] = phi - np.pi - np.arccos((np.square(rou_head3[idt]) + np.square(r) - F_P_square) / (2 * rou_head3[idt] * r))

        ceta_head4 = np.zeros(end_time - int(t3))
        rou_head4 = np.zeros(end_time - int(t3))
        ceta_head4[0] = phi - np.pi
        
        for idt, t in enumerate(np.arange(int(t3) + 1, end_time + 1)):
            t -= t3
            def head_loc(fi):
                m = np.sqrt(1 + np.square(fi))
                return fi * m + np.log(fi + m) - constant - 2 * v * t / gamma
            solve = root(head_loc, ceta_head4[np.max([idt - 1, 0])])
            ceta_t = solve.x[0]
            ceta_head4[idt] = ceta_t
            rou_head4[idt] = ceta_t * gamma
        X_head4 = rou_head4 * np.cos(ceta_head4)
        Y_head4 = rou_head4 * np.sin(ceta_head4)
        ceta_head4 -= np.pi
        X_head4 = - X_head4
        Y_head4 = - Y_head4

        ceta_head = np.append(ceta_head1, ceta_head2)
        ceta_head = np.append(ceta_head, ceta_head3)
        ceta_head = np.append(ceta_head, ceta_head4)

        rou_head = np.append(rou_head1, rou_head2)
        rou_head = np.append(rou_head, rou_head3)
        rou_head = np.append(rou_head, rou_head4)

        X_head = np.append(X_head1, X_head2)
        X_head = np.append(X_head, X_head3)
        X_head = np.append(X_head, X_head4)

        Y_head = np.append(Y_head1, Y_head2)
        Y_head = np.append(Y_head, Y_head3)
        Y_head = np.append(Y_head, Y_head4)

        return ceta_head, rou_head, X_head, Y_head
    
    ceta_head, rou_head, X_head, Y_head = head_locs(start_time, end_time, r)
    # print(x)
    # print(y)
    # print(c)
    # fig = go.Figure()
    # fig.add_scatter(x = X_head[:101], y = Y_head[:101], mode = 'markers')
    # fig.add_scatter(x = X_head[101:101+int(t2)], y = Y_head[101:101 +int(t2)],mode = 'markers')
    # fig.add_scatter(x = X_head[101+int(t2):101+int(t3)], y = Y_head[101+int(t2):101 +int(t3)],mode = 'markers')
    # fig.add_scatter(x = X_head[101 + int(t3):], y = Y_head[101 +int(t3):],mode = 'markers')
    # fig.show()

    def head_and_body(ceta_head, rou_head, X_head, Y_head):
        L1 = 2.86
        during = np.size(ceta_head)

        T_1 = np.intersect1d(np.where(np.square(X_head - x_E) + np.square(Y_head - y_E) <= np.square(L1)), np.where(np.square(X_head - x_A) + np.square(Y_head - y_A) <= np.square(r_A) + 1))
        if np.size(T_1) > 0:
            t_1 = T_1[-1] + 1
        else:
            t_1 = 202
        T_2 = np.where(np.square(X_head - x_C) + np.square(Y_head - y_C) <= np.square(L1))[0]
        if np.size(T_2) > 0:
            t_2 = T_2[-1] + 1
        else:
            t_2 = 202
        T_3 = np.intersect1d(np.where(np.square(X_head - x_F) + np.square(Y_head - y_F) <= np.square(L1)), np.where(ceta_head < phi - np.pi / 2))
        if np.size(T_3) > 0:
            t_3 = T_3[-1] + 1
        else:
            t_3 = 202
        ceta_body1 = np.zeros(during)
        rou_body1 = np.zeros(during)
        X_body1 = np.zeros(during)
        Y_body1 = np.zeros(during)
        # print('>')
        # print(t_1)
        # print(t_4)
        # print(t_5)
        for idt in range(during):
            if idt < t_1:
                def body1_loc1(fi):
                    rou1 = fi * gamma
                    return np.square(rou1) + np.square(rou_head[idt]) - np.square(L1) - 2 * rou1 * rou_head[idt] * np.cos(fi - ceta_head[idt])
                solve = root(body1_loc1, ceta_head[idt] + np.pi / 20)
                ceta_body1[idt] = solve.x[0]
                assert ceta_body1[idt] > ceta_head[idt]
                rou_body1[idt] = ceta_body1[idt] * gamma
                X_body1[idt] = rou_body1[idt] * np.cos(ceta_body1[idt])
                Y_body1[idt] = rou_body1[idt] * np.sin(ceta_body1[idt])
            elif idt < t_2:
                P1_A = np.sqrt(np.square(X_head[idt] - x_A) + np.square(Y_head[idt] - y_A))
                P1_A_P2_cos = (np.square(P1_A) + np.square(r_A) - np.square(L1)) / (2 * P1_A * r_A)
                A_to_H = np.array([X_head[idt] - x_A, Y_head[idt] - y_A]) * r_A * P1_A_P2_cos / P1_A
                H_to_P2 = np.array([y_A - Y_head[idt] , X_head[idt] - x_A]) * r_A * np.sqrt(1 - np.square(P1_A_P2_cos)) / P1_A
                X_body1[idt], Y_body1[idt] = np.array([x_A, y_A]) + A_to_H + H_to_P2
                rou_body1[idt] = np.sqrt(np.square(X_body1[idt]) + np.square(Y_body1[idt]))
                ceta_body1[idt] = ceta_head[idt] + np.arccos((np.square(rou_head[idt]) + np.square(rou_body1[idt]) - np.square(L1)) / (2 * rou_head[idt] * rou_body1[idt]))
            elif idt < t_3:
                B_P1 = np.sqrt(np.square(X_head[idt] - x_B) + np.square(Y_head[idt] - y_B))
                B_P1_P2_cos = (np.square(B_P1) + np.square(L1) - np.square(r_B)) / (2 * B_P1 * L1)
                P1_to_H = np.array([x_B - X_head[idt], y_B - Y_head[idt]]) * (L1 * B_P1_P2_cos / B_P1)
                H_to_P2 = np.array([Y_head[idt] - y_B, x_B - X_head[idt]]) * (L1 * np.sqrt(1 - np.square(B_P1_P2_cos)) / B_P1)
                X_body1[idt], Y_body1[idt] = np.array([X_head[idt], Y_head[idt]]) + P1_to_H + H_to_P2
                rou_body1[idt] = np.sqrt(np.square(X_body1[idt]) + np.square(Y_body1[idt]))
                ceta_body1[idt] = ceta_head[idt] - np.arccos((np.square(rou_head[idt]) + np.square(rou_body1[idt]) - np.square(L1)) / (2 * rou_head[idt] * rou_body1[idt]))
            else:   
                def body1_loc2(fi):
                    rou1 = (fi + np.pi) * gamma
                    return np.square(rou1) + np.square(rou_head[idt]) - np.square(L1) - 2 * rou1 * rou_head[idt] * np.cos(fi - ceta_head[idt])
                solve = root(body1_loc2, ceta_head[idt] - np.pi / 10)
                ceta_body1[idt] = solve.x[0]
                rou_body1[idt] = (ceta_body1[idt] + np.pi) * gamma
                X_body1[idt] = rou_body1[idt] * np.cos(ceta_body1[idt])
                Y_body1[idt] = rou_body1[idt] * np.sin(ceta_body1[idt])   
                assert ceta_body1[idt] < ceta_head[idt]
        # fig = go.Figure()
        # fig.add_scatter(x = X_body1[:int(t_1)], y = Y_body1[:int(t_1)], mode = 'markers')
        # fig.add_scatter(x = X_body1[int(t_1) : int(t_4)], y = Y_body1[int(t_1) : int(t_4)], mode = 'markers')
        # fig.add_scatter(x = X_body1[int(t_4) : int(t_5)], y = Y_body1[int(t_4) : int(t_5)], mode = 'markers')
        # fig.add_scatter(x = X_body1[int(t_5) :], y = Y_body1[int(t_5) :], mode = 'markers')
        # fig.show()
        return ceta_body1, rou_body1, X_body1, Y_body1
    
    ceta_body1, rou_body1, X_body1, Y_body1 = head_and_body(ceta_head, rou_head, X_head, Y_head)
    
    def body_and_body(ceta_body1, rou_body1, X_body1, Y_body1):
        L2 = 1.65
        during = np.size(ceta_body1)
        T_1 = np.intersect1d(np.where(np.square(X_body1 - x_E) + np.square(Y_body1 - y_E) <= np.square(L2)), np.where(np.square(X_body1 - x_A) + np.square(Y_body1 - y_A) <= np.square(r_A) + 1))
        if np.size(T_1) > 0:
            t_1 = T_1[-1] + 1
        else:
            t_1 = 202
        T_2 = np.where(np.square(X_body1 - x_C) + np.square(Y_body1 - y_C) <= np.square(L2))[0]
        if np.size(T_2) > 0:
            t_2 = T_2[-1] + 1
        else:
            t_2 = 202
        T_3 = np.intersect1d(np.where(np.square(X_body1 - x_F) + np.square(Y_body1 - y_F) <= np.square(L2)), np.where(ceta_body1 < phi - np.pi / 2))
        if np.size(T_3) > 0:
            t_3 = T_3[-1] + 1
        else:
            t_3 = 202
        # t_3 = [-1] + 1
        ceta_body2 = np.zeros(during)
        rou_body2 = np.zeros(during)
        X_body2 = np.zeros(during)
        Y_body2 = np.zeros(during)

        for idt in range(during):
            if idt < t_1:
                def body1_loc1(fi):
                    rou1 = fi * gamma
                    return np.square(rou1) + np.square(rou_body1[idt]) - np.square(L2) - 2 * rou1 * rou_body1[idt] * np.cos(fi - ceta_body1[idt])
                solve = root(body1_loc1, ceta_body1[idt] + np.pi / 10)
                # print(solve)
                ceta_body2[idt] = solve.x[0]
                assert ceta_body2[idt] > ceta_body1[idt]
                rou_body2[idt] = ceta_body2[idt] * gamma
                X_body2[idt] = rou_body2[idt] * np.cos(ceta_body2[idt])
                Y_body2[idt] = rou_body2[idt] * np.sin(ceta_body2[idt])
            elif idt < t_2:
                P1_A = np.sqrt(np.square(X_body1[idt] - x_A) + np.square(Y_body1[idt] - y_A))
                P1_A_P2_cos = (np.square(P1_A) + np.square(r_A) - np.square(L2)) / (2 * P1_A * r_A)
                A_to_H = np.array([X_body1[idt] - x_A, Y_body1[idt] - y_A]) * r_A * P1_A_P2_cos / P1_A
                H_to_P2 = np.array([y_A - Y_body1[idt] , X_body1[idt] - x_A]) * r_A * np.sqrt(1 - np.min([1, np.square(P1_A_P2_cos)])) / P1_A
                X_body2[idt], Y_body2[idt] = np.array([x_A, y_A]) + A_to_H + H_to_P2
                rou_body2[idt] = np.sqrt(np.square(X_body2[idt]) + np.square(Y_body2[idt]))
                ceta_body2[idt] = phi - np.arccos((np.square(r) + np.square(rou_body2[idt]) - np.square(X_body2[idt] - x_E) - np.square(Y_body2[idt] - y_E)) / (2 * r * rou_body2[idt]))
            elif idt < t_3:
                B_P1 = np.sqrt(np.square(X_body1[idt] - x_B) + np.square(Y_body1[idt] - y_B))
                B_P1_P2_cos = (np.square(B_P1) + np.square(L2) - np.square(r_B)) / (2 * B_P1 * L2)
                P1_to_H = np.array([x_B - X_body1[idt], y_B - Y_body1[idt]]) * (L2 * B_P1_P2_cos / B_P1)
                H_to_P2 = np.array([Y_body1[idt] - y_B, x_B - X_body1[idt]]) * (L2 * np.sqrt(1 - np.min([1, np.square(B_P1_P2_cos)])) / B_P1)
                X_body2[idt], Y_body2[idt] = np.array([X_body1[idt], Y_body1[idt]]) + P1_to_H + H_to_P2
                rou_body2[idt] = np.sqrt(np.square(X_body2[idt]) + np.square(Y_body2[idt]))
                ceta_body2[idt] = phi - np.pi - np.arccos((np.square(r) + np.square(rou_body2[idt]) - np.square(X_body2[idt] - x_F) - np.square(Y_body2[idt] - y_F)) / (2 * r * rou_body2[idt]))
            else:
                def body1_loc2(fi):
                    rou1 = (fi + np.pi) * gamma
                    return np.square(rou1) + np.square(rou_body1[idt]) - np.square(L2) - 2 * rou1 * rou_body1[idt] * np.cos(fi - ceta_body1[idt])
                solve = root(body1_loc2, ceta_body1[idt] - np.pi / 10)
                ceta_body2[idt] = solve.x[0]
                rou_body2[idt] = (ceta_body2[idt] + np.pi) * gamma
                X_body2[idt] = rou_body2[idt] * np.cos(ceta_body2[idt])
                Y_body2[idt] = rou_body2[idt] * np.sin(ceta_body2[idt])   
                assert ceta_body2[idt] < ceta_body1[idt]
        return ceta_body2, rou_body2, X_body2, Y_body2
    ceta = np.zeros((total_num, 201))
    rou = np.zeros((total_num, 201))
    X = np.zeros((total_num, 201))
    Y = np.zeros((total_num, 201))
    V = np.zeros((total_num, 201))
    ceta[0] = ceta_head
    rou[0] = rou_head
    X[0] = X_head
    Y[0] = Y_head
    ceta[1] = ceta_body1
    rou[1] = rou_body1
    X[1] = X_body1
    Y[1] = Y_body1
    for idx in range(2, total_num):
        print(idx)
        ceta_body2, rou_body2, X_body2, Y_body2 = body_and_body(ceta_body1, rou_body1, X_body1, Y_body1)
        ceta[idx] = ceta_body2
        rou[idx] = rou_body2
        X[idx] = X_body2
        Y[idx] = Y_body2
        ceta_body1, rou_body1, X_body1, Y_body1 = ceta_body2, rou_body2, X_body2, Y_body2
    fig = go.Figure()
    for t in range(201):
        fig.add_scatter(x = [X[0][t], X[1][t], X[2][t]], y = [Y[0][t], Y[1][t], Y[2][t]], mode = 'markers+lines')    
    fig.show()

    fig = go.Figure()
    for t in np.arange(0, 210, 10):
        fig.add_scatter(x = X.T[t], y = Y.T[t], mode = 'markers+lines')  
    fig.show()

    def Ciallo(point):
        ceta, rou, x, y = point
        if np.square(x) + np.square(y) >= np.square(r):
            if np.abs(rou / gamma - ceta) < np.pi / 2:
                return np.array([(np.cos(ceta) - ceta * np.sin(ceta)) / np.sqrt(1 + np.square(ceta)), (np.sin(ceta) + ceta * np.cos(ceta)) / np.sqrt(1 + np.square(ceta))])
            else:
                return np.array([(np.cos(ceta) - (ceta + np.pi) * np.sin(ceta)) / np.sqrt(1 + np.square(ceta + np.pi)), (np.sin(ceta) + (ceta + np.pi) * np.cos(ceta)) / np.sqrt(1 + np.square(ceta + np.pi))])
        else:
            if np.dot(np.array([x_E, y_E]), np.array([x - x_C, y - y_C])) >= 0:
                return np.array([(y_A - y), (x - x_A)]) / r_A
            else:
                return np.array([(y_B - y), (x - x_B)]) / r_B
                
    def velocity(point1, point2, v1):
        ceta1, rou1, x1, y1 = point1
        ceta2, rou2, x2, y2 = point2
        pwp = Ciallo(point1)
        qwq = Ciallo(point2)
        QAQ = np.array([x1 - x2, y1 - y2])
        v2 = v1 * np.abs(np.dot(pwp, QAQ) / np.dot(qwq, QAQ))
        return v2

    for idt in range(201):
        V[0][idt] = v
        
        V[1][idt] = velocity(np.array([ceta[0][idt], rou[0][idt], X[0][idt], Y[0][idt]]), np.array([ceta[1][idt], rou[1][idt], X[1][idt], Y[1][idt]]), V[0][idt])
        for idx in np.arange(1, total_num - 1):
            V[idx + 1][idt] = velocity(np.array([ceta[idx][idt], rou[idx][idt], X[idx][idt], Y[idx][idt]]), np.array([ceta[idx + 1][idt], rou[idx + 1][idt], X[idx + 1][idt], Y[idx + 1][idt]]), V[idx][idt])
    # print(V[0])
    # print(V[1])
    # print(V[2])
    # print(V[100])
    # print(V[-1])
    fig = go.Figure()
    fig.add_scatter(x = np.arange(201), y = V[0], mode = 'markers+lines')
    fig.add_scatter(x = np.arange(201), y = V[1], mode = 'markers+lines')
    fig.add_scatter(x = np.arange(201), y = V[2], mode = 'markers+lines')
    fig.show()
    print(X)
    print(Y)
    print(V)
    # return ceta, rou, X, Y, V


    print(f'rA: {r_A}')
    print(f'rB: {r_B}')
    print(f'大圆弧: {r_A * (np.pi - 2 * alpha)}')
    print(f'小圆弧: {r_B * (np.pi - 2 * alpha)}')


    # start_time = t3
    print('>>')
    print(t3)
    t_3 = t3 + 100
    end_time = 115
    time_gap = 1000
    start_time = end_time - int((end_time - t_3) * time_gap) / time_gap 
    t_span = int((end_time - t_3) * time_gap)
    print(start_time)
    # fig = go.Figure()
    # fig.add_scatter(x = X.T[113], y = Y.T[113], mode = 'markers+lines')
    # fig.add_scatter(x = X.T[114], y = Y.T[114], mode = 'markers+lines')
    # fig.add_scatter(x = X.T[115], y = Y.T[115], mode = 'markers+lines')
    # fig.add_scatter(x = X.T[116], y = Y.T[116], mode = 'markers+lines')
    # fig.show()

    medium = np.sqrt(1 + np.square(phi))
    constant = (phi) * medium + np.log(phi + medium)
    ceta_v = np.zeros(t_span + 1)
    rou_v = np.zeros(t_span + 1)
    x_v = np.zeros(t_span + 1)
    y_v = np.zeros(t_span + 1)
    vs2 = np.zeros(t_span + 1)
    vs3 = np.zeros(t_span + 1)
    vs4 = np.zeros(t_span + 1)
    for idt, t in enumerate(np.arange(start_time, end_time, 1 / time_gap)):
        t -= start_time
        def head_loc(fi, t = t):
            m = np.sqrt(1 + np.square(fi))
            return fi * m + np.log(fi+ m) - constant - 2 * v * t / gamma
        # print(t)
        solve = root(head_loc, phi + np.pi / 10)
        ceta_t = solve.x[0]
        ceta_v[idt] = ceta_t
        rou_v[idt] = ceta_t * gamma
    x_v = rou_v * np.cos(ceta_v)
    y_v = rou_v * np.sin(ceta_v)
    ceta_v -= np.pi
    x_v = - x_v
    y_v = - y_v

    ceta_v2, rou_v2, x_v2, y_v2 = head_and_body(ceta_v, rou_v, x_v, y_v)
    ceta_v3, rou_v3, x_v3, y_v3 = body_and_body(ceta_v2, rou_v2, x_v2, y_v2)
    ceta_v4, rou_v4, x_v4, y_v4 = body_and_body(ceta_v3, rou_v3, x_v3, y_v3)
    # print(ceta_v, ceta_v2)
    # print(rou_v, rou_v2)
    # print(x_v, x_v2)
    # print(y_v, y_v2)
    
    try:
        for idt, t in enumerate(np.arange(start_time, end_time, 1 / time_gap)):
            vs2[idt] = velocity(np.array([ceta_v[idt], rou_v[idt], x_v[idt], y_v[idt]]), np.array([ceta_v2[idt], rou_v2[idt], x_v2[idt], y_v2[idt]]), 1)
    except Exception as e:
        print(e)
    for idt, t in enumerate(np.arange(start_time, end_time, 1 / time_gap)):
        vs3[idt] = velocity(np.array([ceta_v2[idt], rou_v2[idt], x_v2[idt], y_v2[idt]]), np.array([ceta_v3[idt], rou_v3[idt], x_v3[idt], y_v3[idt]]), 1)
    vs3 *= vs2
    for idt, t in enumerate(np.arange(start_time, end_time, 1 / time_gap)):
        vs4[idt] = velocity(np.array([ceta_v3[idt], rou_v3[idt], x_v3[idt], y_v3[idt]]), np.array([ceta_v4[idt], rou_v4[idt], x_v4[idt], y_v4[idt]]), 1)
    vs4 *= vs3
    fig = go.Figure()
    fig.add_scatter(x = np.arange(start_time, end_time, 1 / time_gap), y = vs2, mode = 'markers+lines', name = '第二个孔的速度倍率')
    fig.add_scatter(x = np.arange(start_time, end_time, 1 / time_gap), y = vs3, mode = 'markers+lines', name = '第三个孔的速度倍率')
    fig.add_scatter(x = np.arange(start_time, end_time, 1 / time_gap), y = vs4, mode = 'markers+lines', name = '第四个孔的速度倍率')
    fig.show()
    # fig = go.Figure()
    # fig.add_scatter(x = x_v, y = y_v, mode = 'markers')
    # # fig.add_scatter(x = x_v2, y = y_v2, mode = 'markers')
    # fig.show()
# Q4()
