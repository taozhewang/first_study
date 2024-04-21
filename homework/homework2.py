import numpy as np
import scipy 
from scipy.stats import norm
from scipy.integrate import quad
import scipy.stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import root
import timeit
# mu = 0.5
# sigma = 0.2
# RV = norm(loc = mu, scale = sigma)

def expectation(RV, k):
    def f(x):
        pdf = RV.pdf
        return ((x - k) ** 2) * pdf(x)
    return quad(f, -np.inf, np.inf)
# print('1.1:', expectation(RV, 2))

def mc_expectation(RV, k, N):
    I = np.ones(N)############################
    for j in range(1, N + 1):##############    X = RV.rvs(N)
        i = 0###################################
        for _ in range(j):
            x = scipy.stats.norm.rvs(loc = mu, scale = sigma)
            i += (x - k) ** 2
        i = i / j
        I[j - 1] = i
    return I
# print('1.2:', mc_expectation(RV, 2, 10))

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
# plot_mc_convergence(RV, 2, 100)

# Quantile
def quantile00(RV, u = 0.5):
    assert u < 1 and u > 0, '0 < u < 1'
    cdf = RV.cdf
    return root(cdf, u)
# print('2.1:', quantile00(RV, 0.5))

# def compare_speed_nm(u):
#     nm = scipy.stats.norm(0, 1)
    
    
#     return (timeit.timeit('quantile00(nm, u)', number = 1000, globals = globals()),
#              timeit.timeit(nm.ppf(u))), (timeit.timeit('quantile00(nm, u)', number = 1000, globals = globals()), 
#                                   timeit.timeit(st.ppf(u))), (('quantile00(nm, u)', number = 1000, globals = globals()),
#                                                         timeit.timeit(ig.ppf(u)))
def compare_speed_st(u):
    RV = scipy.stats.t(2)
    def quantile00(u):
        cdf = RV.cdf
        return root(cdf, u)
    return timeit.timeit('quantile00(u)', number = 1000, globals = globals())
print(compare_speed_st(0.5))
# def compare_speed_ig(u):
#     ig = scipy.stats.norminvgauss(0, 1)
# print(compare_speed(0.5))

def f(x):
    return 3 * x
def time():
    return timeit.timeit('f(3)', number = 1000, globals = globals())
print(time())