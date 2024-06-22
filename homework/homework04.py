import numpy as np
import plotly.graph_objects as go

# ODE implementation
# x" + ω²x = 0 -> x' - v = 0 ; ω²x + v' = 0
omiga = 1
def v(x):
    return - x * pow(omiga, 2)
def x(v):
    return v
def solve2dim(v, x, x0, v0, bound, n):
    X = np.zeros(n)
    V = np.zeros(n)

    X[0] = x0
    V[0] = v0
    space = bound / n
    for i in range(n - 1):
        X[i + 1] = X[i] + space * x(V[i])
        V[i + 1] = V[i] + space * v(X[i])

    fig = go.Figure()
    fig.add_scatter3d(x = np.linspace(0, bound, n), y = X, z = V, mode = 'markers')
    fig.show()
    return

solve2dim(v, x, 1, 1, 10, 100)

# x" + 2γx' + ω²x = 0 -> x' = v ; v' = -2γv - ω²x
omiga = 1
gama = 0.1
def x(v):
    return v
def v(x, v):
    return - 2 * gama * v - pow(omiga, 2) * x
def solve2dim2(v, x, x0, v0, bound, n):
    X = np.zeros(n)
    V = np.zeros(n)

    X[0] = x0
    V[0] = v0
    space = bound / n
    for i in range(n - 1):
        x1 = x(V[i])
        v1 = v(X[i], V[i])
        x2 = x(V[i] + space * v1 / 2)
        v2 = v(X[i] + space * x1 / 2, V[i] + space * v1 / 2)
        x3 = x(V[i] + space * v2 / 2)
        v3 = v(X[i] + space * x2 / 2, V[i] + space * v2 / 2)
        x4 = x(V[i] + space * v3)
        v4 = v(X[i] + space * x3, V[i] + space * v3)
                
        X[i + 1] = X[i] + space * (x1 + 2 * x2 + 2 * x3 + x4) / 6
        V[i + 1] = V[i] + space * (v1 + 2 * v2 + 2 * v3 + v4) / 6

    fig = go.Figure()
    fig.add_scatter3d(x = np.linspace(0, bound, n), y = X, z = V, mode = 'markers')
    fig.show()
    return

solve2dim2(v, x, 1, 1, 10, 100)

# Forward vs Backward (explicit vs implicit)
import scipy as sp
from scipy.optimize import root
#explicit
l = 1
def y(t):
    return - l * t
def explicit(y, y0, bound, n):
    space = bound / n
    T = np.linspace(0, bound, n)
    Y = np.zeros(n)

    Y[0] = y0
    for i in range(n - 1):
        Y[i + 1] = Y[i] + space * y(Y[i])
    return Y
#RK 4th
def RK_4(y, y0, bound, n):
    space = bound / n
    T = np.linspace(0, bound, n)
    Y = np.zeros(n)
    Y[0] = y0
    for i in range(n - 1):
        k1 = y(Y[i])
        k2 = y(Y[i] + space * k1 / 2)
        k3 = y(Y[i] + space * k2 / 2)
        k4 = y(Y[i] + space * k3)
        Y[i + 1] = Y[i] + space * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Y        

def implicit(y, y0, bound, n):
    space = bound / n
    T = np.linspace(0, bound, n)
    Y = np.zeros(n)
    Y[0] = y0
    for i in range(n - 1):
        def f(x):
            return y(x) - x + Y[i]

        Y[i + 1] = root(f, 0).x
    return Y
def solution(y0, bound, n):
    T = np.linspace(0, bound, n)
    Y = np.zeros(n)
    Y[0] = y0
    for i in range(n):
        Y[i] = y0 * np.exp(- l * T[i])
    return Y
Y1 = explicit(y, 1, 10, 100)
Y2 = RK_4(y, 1, 10, 100)
Y3 = implicit(y, 1, 10, 100)
Y4 = solution(1, 10, 100)
fig = go.Figure()
fig.add_scatter(x = np.linspace(0, 10, 100), y = Y1, mode = 'markers', name = 'explicit')
fig.add_scatter(x = np.linspace(0, 10, 100), y = Y2, mode = 'markers', name = 'RK 4th')
fig.add_scatter(x = np.linspace(0, 10, 100), y = Y3, mode = 'markers', name = 'implicit')
fig.add_scatter(x = np.linspace(0, 10, 100), y = Y4, mode = 'markers', name = 'solution')
fig.show()

import timeit
a = timeit.timeit(f'explicit(y,{1},{10},{100})', number = 0, globals = globals())
b = timeit.timeit(f'RK_4(y,{1},{10},{100})', number = 0, globals = globals())
c = timeit.timeit(f'implicit(y,{1},{10},{100})', number = 0, globals = globals())
print(f'time \n explicit: {a}, RK 4th: {b}, implicit: {c}')

#3. Gradient Descent
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split

digits, target = dt.load_digits(n_class = 2, return_X_y = True)
print(digits.shape)
px.imshow(digits.reshape(360, 8, 8)[:10, :, :], facet_col = 0, binary_string = True)
x_train, x_test, y_train, y_test = train_test_split(digits, target, test_size = 0.2, random_state = 10)
print(x_train, y_train)
def objectfunc(W, x_train, y_train):
    row = np.size(x_train, 0)
    one = np.ones((row, 1))
    X = np.append(one, x_train, axis =1)
    P = y_train - W.T @ X.T
    result = P @ P.T
    return result / len(y_train)
def gradient(W, x_train, y_train):
    row = np.size(x_train, 0)
    one = np.ones((row, 1))
    X = np.append(one, x_train, axis =1)
    P = y_train - W.T @ X.T
    w_gradient = np.zeros(len(W))
    for i in range(len(W)):
        w_gradient[i] = - 2 * P @ X[:, i]
    return w_gradient / len(y_train)
def gradient_descent(w0, x_train, y_train, learning_rate, momentum):
    max_iter = 100
    treshold = 1e-02
    i = 0
    dw = np.zeros((len(w0), max_iter))
    w = np.zeros((len(w0), max_iter))
    obj = np.zeros(max_iter)
    w[:, 0] = w0
    obj[0] = objectfunc(w[:, 0], x_train, y_train)
    dw[:, 0] = 0
    diff = 1

    while i < max_iter - 1 and diff > treshold:
        dw[:, i + 1] = momentum * dw[:, i] + learning_rate * gradient(w[:, i], x_train, y_train)
        w[:, i + 1] = w[:, i] - dw[:, i + 1]
        obj[i + 1] = objectfunc(w[:, i + 1], x_train, y_train)
        diff = np.abs(obj[i + 1] - obj[i])
        # print(obj[i], w[:, i])
        i += 1
    return w, obj
def accuracy(W, x_test, y_test):
    row = np.size(x_test, 0)
    one = np.ones((row, 1))
    X = np.append(one, x_test, axis =1)
    P = W.T @ X.T
    k = 0
    for i, p in enumerate(P):
        k += bool(bool(p >= 0.5) == y_test[i])
        m = i + 1
    return k / m

w0 = np.ones(np.size(x_train, axis = 1) + 1) * 0
learning_rate = 3e-04
momentum = 1e-01
w, obj = gradient_descent(w0, x_train, y_train, learning_rate, momentum)
obj = obj[obj != 0]
w = w[:, : len(obj)]
print(f'w: {w} \n\n obj: {obj}')

p_train = accuracy(w[:, len(obj) - 1], x_train, y_train)
p_test = accuracy(w[:, len(obj) - 1], x_test, y_test)
print(f'accuracy of training set: {p_train}')
print(f'accuracy of test set: {p_test}')
