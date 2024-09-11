import numpy as np
import scipy as sc
from scipy.optimize import root
import plotly.graph_objects as go
from scipy.optimize import minimize
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
    return X, Y
X, Y = question2(0, 5, 100)
fig = go.Figure()
fig.add_scatter(x = X[0], y = Y[0], mode = 'lines')
fig.show()
print()