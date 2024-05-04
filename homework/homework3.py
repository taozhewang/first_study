import numpy as np
# Q1
def find_root(f, err):
    left_bound = -1
    right_bound = 1
    range_of_root = right_bound - left_bound
    err = err * 2
    while range_of_root >= err:
        left_value = f(left_bound)
        right_value = f(right_bound)
        middle_bound = (left_bound + right_bound) / 2
        assert left_value * right_value < 0, 'please change the bound of root'
        middle_value = f(middle_bound)
        if middle_value == 0:
            print(f'one root of f is {(left_bound + right_bound) / 2}, error = 0')
            return None
        elif middle_value * left_value < 0:
            right_bound = middle_bound
        else:
            left_bound = middle_bound
        range_of_root = range_of_root / 2
    print(middle_value)
    return None
# find_root(lambda x : 3 * pow(x, 3) + 2 * pow(x, 2) + x + 1, 0.001)

# Q2
def find_min(X, Y):
    assert len(X) == len(Y), 'X and Y must have the same length'
    n = len(X)
    sigma_X = np.sum(X)
    sigma_Y = np.sum(Y)
    sigma_square_X = np.dot(X, X)
    sigma_XY = np.dot(X, Y)
    coefficient_matrix = np.array([[   n   ,    sigma_X    ], 
                                   [sigma_X, sigma_square_X]])
    vector = np.array([sigma_Y, sigma_XY])
    solution = np.linalg.solve(coefficient_matrix, vector)
    print(f'beta0 is: {solution[0]}; beta1 is: {solution[1]}')
    return None
# X = np.random.random(10)
# Y = np.random.random(10)
# print(X, Y, sep = '\n')
# find_min(X, Y)

# Q3
import plotly.graph_objects as go
def initial_value_problem(f, y_0, dt):
    assert dt > 0, 'dt must be larger than 0'
    t = 0
    Y = np.array([y_0], dtype = np.float16)
    T = np.array([0], dtype = np.float16)
    y_t = y_0
    for _ in range(100):
        y_t += f(y_t, t) * dt
        t += dt
        Y = np.append(Y, y_t)
        T = np.append(T, t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = T, 
                             y = Y, 
                                 mode = 'markers + lines', 
                                 name = 'y(t)'))
    fig.update_layout(xaxis_title = 't Axis', 
                      yaxis_title = 'y Axis', 
                      title = 'y(t)')
    fig.show()
# def f(y, t):
#     return y + t
# initial_value_problem(f, 1, 0.1)
    
# Q4
from scipy.optimize import minimize
import timeit
def integral(f, left_bound, right_bound, n = 1000):
    X = np.linspace(left_bound, right_bound, num = n)
    Y = f(X)
    print(np.sum(Y) * (right_bound - left_bound) / n)
    return None
def MC_integral(f, left_bound, right_bound, n = 1000):
    x0 = (left_bound + right_bound) / 2
    result1 = minimize(f, x0, method='Nelder-Mead', options={'disp': True})
    result2 = minimize(lambda x: -f(x), x0, method='Nelder-Mead', options={'disp': True})
    upper_bound = -result2.fun
    lower_bound = result1.fun
    X = np.random.random(n)
    X = np.ones(n) * left_bound + X * (right_bound - left_bound)
    Y = np.random.random(n)
    Y = np.ones(n) * lower_bound + Y * (upper_bound - lower_bound)
    N = np.sum(Y < f(X))    
    integration = (N / n) * ((right_bound - left_bound) * (upper_bound - lower_bound)) - (right_bound - left_bound) * (0 - lower_bound)
    print(integration)
    return None

def f(x):
    return np.sin(x)
# left_bound = 0
# right_bound = (3 / 2) * np.pi
# n = 100000
# integral(f, left_bound, right_bound, n)
# MC_integral(f, left_bound, right_bound, n)
# print(timeit.timeit(f'integral(f, {left_bound}, {right_bound}, {n})', number = 0, globals = globals()),
#        timeit.timeit(f'MC_integral(f, {left_bound}, {right_bound}, {n})', number = 0, globals = globals()))

# Q5
def Kth_number(X, k):
    assert k <= len(X), 'k must be lower than len(X)'
    Y = np.sort(X)
    print(Y)
    print(Y[len(X) - k])
    return None
# X = np.random.randint(20, size = 10)
# np.random.shuffle(X)
# Kth_number(X, 5)

# Q6
def remove_average(X, k):
    assert k <= len(X), 'k must be lower than len(X)'
    Y = np.sort(X)
    print(Y)
    Y = np.delete(Y, [k - 1, len(X) - k])
    print(Y)
    return None
# X = np.random.randint(20, size = 10)
# np.random.shuffle(X)
# remove_average(X, 3)

# Q7
def gate():
    n = 1000
    door = np.array([1, 0, 0])
    shift = 0
    for i in range(n):
        choice = np.random.randint(3)
        place = np.where(door == 0)
        p = np.array([])
        for j in place[0]:
            if j != choice:
                p = np.append(p, j)
        host = int(np.random.choice(p))
        curr_door = np.delete(door, [choice, host])
        if curr_door == 1:
            shift += 1
    print('shift:', shift, 'not shift:', n - shift, sep = '\n')
    if shift > n - shift:
        print('shifting is better')
    else:
        print('not shifting is better')
# gate()

# Q8
def move(loc):
    if loc == 1:
        return np.array([1/2, 1/2, 0])
    elif loc == 2:
        return np.array([1/2, 1/4, 1/4])
    else:
        return np.array([0, 1/3, 2/3])
def multiple_move(n):
    loc = 1
    for _ in range(n):
        loc_p = move(loc)
        loc = np.random.choice([1, 2, 3], p = loc_p)
    return loc
def MC_loc():
    n = 1000
    steps = 1000
    loc_statistics = [0, 0, 0]
    for _ in range(n):
        loc_statistics[multiple_move(steps) - 1] += 1
    print(loc_statistics)
    return None
# MC_loc()

# Q9
def clt(n):
    X = np.random.random(n)
    a = np.sum(X) - n * (1/2)
    b = np.sqrt(n * (1/12))
    return a / b
def plot_clt():
    N = np.arange(1, 1000000, 1000)
    Y = [clt(n) for n in N]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = N, 
                             y = Y, 
                                 mode = 'markers + lines', 
                                 name = 'clt(n)'))
    fig.update_layout(xaxis_title = 'n Axis', 
                      yaxis_title = 'y Axis', 
                      title = 'Central Limit Theorem')
    fig.show()
# plot_clt()

