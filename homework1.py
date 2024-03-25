# Datatypes, Control Flows, Functions
def count_five_or_seven():
    for i in range(1000):
        if (i + 1) % 5 ==0 or (i + 1) % 7 == 0:
            print(i)
def print_S0():
    def row():
        for _ in range(3):
            print('o' * 17)
    def column(a):
        row()
        if a:
            for _ in range(3):
                print(' ' * 13, 'o'* 4, sep = '')
            row()
        else:
            for _ in range(3):
                print('o'* 4)
    for i in range(2):
        column(i)
def search_vegetable(vegetfruit):
    vegetable = {}
    for i in range(len(vegetfruit)):
        if vegetfruit[i] == "cabage" or "potato":
            vegetable[i] = vegetfruit[i]
    return vegetable
def inverse_list(your_list):
    a = []
    for i in range(len(your_list) - 1, -1, -1):
        a.append(your_list[i])
    return a
def days_month(month):
    month = month.strip().lower()
    year = {'january': 31, 'february': 28, 'march': 31, 'april': 30, 'may': 31, 'june': 30,
            'july': 31, 'august': 31, 'september': 30, 'october': 31, 'november': 30, 'december':31}
    return year[month]
def order_list(num):
    length = len(num)
    for i in range(length):
        for j in range(length - i - 1):
            if num[j] > num[j + 1]:
                num[j], num[j + 1] = num[j + 1], num[j]
    return(num)

# Numpy
import numpy as np

def division():
    a = np.arange(0, 1001)
    b = (a % 5 == 0) + (a % 7 == 0) ^ (a == 0)
    return np.where(b)
def sort(num):
    return np.sort(num)
def nd(n, d):
    return np.arange(n * d).reshape(n, d)
def permutation():
    a = np.arange(1, 10)
    np.random.shuffle(a)
    for _ in range(8):
        b = np.arange(1, 10)
        np.random.shuffle(b)
        a = np.concatenate((a, b))
    return a.reshape((9, 9))
def normalize(a):
    return a / np.linalg.norm(a)
def round(a):
    return np.round(a, 2)
def closest_to_zero_point_six(a):
    a = abs(a - 0.6)
    closest = np.min(a)
    location = np.where(a == closest)
    return closest, location
def histogram(a):
    numbers = []
    for i in range(10):
        i = i / 10
        p = np.where((a < i + 0.1) * (a >= i))
        l = p[0]
        numbers.append(len(l))
    return numbers

# Numpy and Plotly: Collatz Conjecture
import plotly.graph_objects as go

def compute(n):
    b = np.array([n])
    while True:
        if n == 1:
            return b
        elif n % 2 == 0:
            n = n / 2
        else:
            n = 3 * n + 1
        b = np.append(b, n)

def plot_path(a):
    paths = []
    l = []
    for i in a:
        b = compute(i)
        paths.append(b)
        l.append(len(b))
    length = max(l)
    fig = go.Figure()
    for i in range(len(paths)):
        fig.add_trace(go.Scatter(x = [i for i in range(1, length + 1)], 
                                 y = paths[i], 
                                 mode = 'markers + lines', 
                                 name = f'Group {i+1}'))
    fig.update_layout(xaxis_title = 'X Axis', 
                      yaxis_title = 'Y Axis', 
                      title = 'paths')
    fig.show()

def compute_step(a):
    a_steps = np.array([])
    for i in a:
        step = 0
        while True:
            if i == 1:
                a_steps = np.append(a_steps, step)
                break
            elif i % 2 == 0:
                i, step = i / 2, step + 1
            else:
                i, step = 3 * i + 1, step + 1
    return a_steps

def plot_bar(N):
    assert N <= 10000, 'N must not be bigger than 10000'
    a = np.arange(1, N + 1)
    b = compute_step(a)
    fig = go.Figure()
    fig.add_bar(x = a, y = b)
    fig.update_layout(title = 'histogram', xaxis_title = 'number', yaxis_title = 'length')
    fig.show()

# Scipy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
def plot1():
    x = np.linspace(-10, 10, 100) 
    y = x ** 2 + 10 * np.sin(x)
    plt.plot(x, y)
    plt.title('Plot of f(x) = x^2+10sin(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()
def find_min():
    y = lambda x : x ** 2 + 10 * np.sin(x)
    a = minimize(y, x0 = -1)
    return a

import scipy.stats as stats
from scipy.stats import norm
def normal(k = 1):
    x = np.linspace(-5 * k, 5 * k, 100 * k)
    y_p = norm.pdf(x, 0, k)
    y_c = norm.cdf(x, 0, k)
    plt.plot(x, y_p, label = 'pdf')
    plt.plot(x, y_c, label = 'cdf')
    plt.legend()
    plt.title('Plot of pdf and cdf')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
def three_normal():
    for i in [1, 2, 5]:
        normal(i)

def freedom(f = 2):
    x = np.linspace(-5, 5, 100)
    distribution = stats.t(f)
    y_p = distribution.pdf(x)
    y_c = distribution.cdf(x)
    plt.plot(x, y_p, label = 'pdf')
    plt.plot(x, y_c, label = 'cdf')
    plt.legend()
    plt.title(f'Plot of pdf and cdf in {f} freedom')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
def four_freedom():
    for i in [2, 3, 4, 6]:
        freedom(i)
