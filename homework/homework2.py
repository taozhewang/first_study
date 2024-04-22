import numpy as np
import scipy 
from scipy.stats import norm
from scipy.integrate import quad
import scipy.stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize

import timeit
mu = 0.5
sigma = 0.2
RV = norm(loc = mu, scale = sigma)

def expectation(RV, k):
    def f(x):
        pdf = RV.pdf
        return (np.maximum((x - k), 0)) * pdf(x)
    return quad(f, -np.inf, np.inf)
# print('1.1:', expectation(RV, 0.4))

def mc_expectation(RV, k, N):
    I = np.ones(N)
    for j in range(1, N + 1):
        X = RV.rvs(j)
        I[j - 1] = sum(np.maximum((X - k), 0)) / j
    return I
# print('1.2:', mc_expectation(RV, 0.4, 20))

def plot_mc_convergence(RV, k, N):
    x = np.arange(N)
    y_ex = np.ones(N) * expectation(RV, k)[0]
    plt.plot(x, y_ex, label = 'expectation')
    for _ in range(10):
        y_mc_ex = mc_expectation(RV, k, N)
        plt.plot(x, y_mc_ex, label = 'mc_expectation')
    plt.legend()
    plt.title('Convergence of Monte Carlo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
# plot_mc_convergence(RV, 0.4, 1000)

# Quantile
def quantile00(RV, u = 0.5):
    assert u < 1 and u > 0, '0 < u < 1'
    cdf = RV.cdf
    return root(cdf, u)

def compare_speed_nm(u):
    return (timeit.timeit(f'quantile00(scipy.stats.norm(0.5, 0.2), {u})', number = 1000, globals = globals()),
             timeit.timeit(f'scipy.stats.norm(0.5, 0.2).ppf({u})', number = 1000, globals = globals()))
def compare_speed_st(u):
    return (timeit.timeit(f'quantile00(scipy.stats.t(2), {u})', number = 1000, globals = globals()),
             timeit.timeit(f'scipy.stats.t(2).ppf({u})', number = 1000, globals = globals()))
def compare_speed_ig(u):
    return (timeit.timeit(f'quantile00(scipy.stats.norminvgauss(0.5, 0.2), {u})', number = 1000, globals = globals()),
             timeit.timeit(f'scipy.stats.norminvgauss(0.5, 0.2).ppf({u})', number = 1000, globals = globals()))
# print(compare_speed_nm(0.5))
# print(compare_speed_st(0.5))
# print(compare_speed_ig(0.5))
def empirical(N, u):
    def em_quantile(N, u):
        Y = np.ones(N - 1)
        for i in range(1, N):
            X = scipy.stats.norm(0.5, 0.2).rvs(N)
            y = np.quantile(X, u)
            Y[i - 1] = y
        return Y
    x = np.arange(1, N)
    Y = em_quantile(N, u)
    theo = RV.ppf(u)
    y_t = np.ones(N - 1) * theo
    plt.plot(x, Y, label = 'empirical quantile')
    plt.plot(x, y_t, label = 'theoretical quantile')
    plt.legend()
    plt.title('Convergence of empirical quantile')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
# empirical(1000, 0.5)

# Average Value at Risk
alpha = 0.99
def AVaR0(RV, alpha):
    return quad(RV.ppf, 1 - alpha, 1)
def AVaR1(RV, alpha):
    def f(x):
        def g(y):
            return (np.maximum((y - x), 0)) * RV.pdf(y)
        result, err = quad(g, -np.inf, np.inf)
        return result / alpha + x
    return minimize(f, x0 = 0)
def AVaR2(RV, alpha):
    def f(x):
        return np.maximum((x - RV.ppf(1 - alpha)), 0) * RV.pdf(x)
    result, err =quad(f, -np.inf, np.inf)
    return RV.ppf(1 - alpha) + result / alpha
# a1 = AVaR0(RV, alpha)
# a2 = AVaR1(RV, alpha)
# a3 = AVaR2(RV, alpha)
# print(a1, a2, a3, sep = '\n')

N = 1000
X = RV.rvs(N)
def mc_AVaR0(X, alpha):
    def f(x):
        return np.quantile(X, x)
    result, err = quad(f, 1- alpha, alpha)
    return result / alpha, err / alpha
def mc_AVaR1(X, alpha):
    def f(x):
        return x + sum(np.maximum((X - x), 0)) / (len(X) * alpha)
    return minimize(f, x0 = 0)
def mc_AVaR2(X, alpha):
    def f(x):
        N = 1000
        X = RV.rvs(N)
        return np.quantile(X, x)
    return f(1 - alpha) + sum(np.maximum((X - f(1 - alpha)), 0)) / (len(X) * alpha)
# ma1 = mc_AVaR0(np.random.random(N), alpha)
# ma2 = mc_AVaR1(np.random.random(N), alpha)
# ma3 = mc_AVaR2(np.random.random(N), alpha)
# print(ma1, ma2, ma3, sep = '\n')

def speed():
    print(timeit.timeit(f'AVaR0(scipy.stats.norm(0.5, 0.2), {alpha})', number = 1000, globals = globals()))
    print(timeit.timeit(f'AVaR1(scipy.stats.norm(0.5, 0.2), {alpha})', number = 1000, globals = globals()))
    print(timeit.timeit(f'AVaR2(scipy.stats.norm(0.5, 0.2), {alpha})', number = 1000, globals = globals()))
    print(timeit.timeit(f'mc_AVaR0(np.random.random({N}), {alpha})', number = 1000, globals = globals()))
    print(timeit.timeit(f'mc_AVaR1(np.random.random({N}), {alpha})', number = 1000, globals = globals()))
    print(timeit.timeit(f'mc_AVaR2(np.random.random({N}), {alpha})', number = 1000, globals = globals()))
# speed()

# multidymensional
from scipy.stats import multivariate_normal
# case 2
mu = np.array([0.2, 0.5])
Sigma = np.array([[1.         , -0.26315789], 
                  [-0.26315789, 1.         ]])
w = np.array([2, 5])
RV_2dim = multivariate_normal(mean = mu, cov = Sigma)
N = 100
samples = RV_2dim.rvs(N)
X = samples @ w.T
def mc_AVaR0_dim2(X, alpha):
    def f(y):
        return np.quantile(X, y)
    result, err = quad(f, 1 - alpha, 1)
    return result / alpha, err / alpha
# print(mc_AVaR0_dim2(X, alpha))
def mc_AVaR1_dim2(X, alpha):
    def f(x):
        return x + sum(np.maximum((X - x), 0)) / (alpha * len(X))
    return minimize(f, x0 = 0.5)
# print(mc_AVaR1_dim2(X, alpha))
def mc_AVaR2_dim2(X, alpha):
    return np.quantile(X, 1 - alpha) + (sum(np.maximum(X - np.quantile(X, 1 - alpha), 0))) / (alpha * len(X))
# print(mc_AVaR2_dim2(X, alpha))
#case 5
mu = np.array([0.2, 0.5, -0.1, 0, 0.6])
Sigma = np.array([[1.        , 0.2688825 , 0.401427  , 0.19473116, 0.66256879],
                  [0.2688825 , 1.        , 0.3907619 , 0.43373298, 0.43199657],
                  [0.401427  , 0.3907619 , 1.        , 0.27893741, 0.61330745],
                  [0.19473116, 0.43373298, 0.27893741, 1.        , 0.46849892],
                  [0.66256879, 0.43199657, 0.61330745, 0.46849892, 1.        ]])
w = np.array([2, 5, -2, 3, 6])
RV_5dim = multivariate_normal(mean = mu, cov = sigma)
N = 100
samples =RV_5dim.rvs(N)
X = np.dot(samples, w)
def mc_AVaR0_dim5(X, alpha):
    def f(y):
        return np.quantile(X, y)
    result, err = quad(f, 1 - alpha, 1)
    return result / alpha, err / alpha
# print(mc_AVaR0_dim2(X, alpha))
def mc_AVaR1_dim2(X, alpha):
    def f(x):
        return x + sum(np.maximum((X - x), 0)) / (alpha * len(X))
    return minimize(f, x0 = 0.5)
# print(mc_AVaR1_dim2(X, alpha))
def mc_AVaR2_dim2(X, alpha):
    return np.quantile(X, 1 - alpha) + (sum(np.maximum(X - np.quantile(X, 1 - alpha), 0))) / (alpha * len(X))
# print(mc_AVaR2_dim2(X, alpha))