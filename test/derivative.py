import numpy as np
import timeit
import plotly.graph_objects as go
# f = lambda x: np.e ** (np.sin(x))
f = lambda x: np.exp(x + x ** 2)
x = 0
# h = 1e-04
def for_der(f, x, h):
    return (f(x + h) - f(x)) / h
def bac_der(f, x, h):
    return (-f(x - h) + f(x)) / h
def gra_der(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# N = np.linspace(-5, 0, 100)
N = np.linspace(-18, 0, 1000)
h = 10 ** N
df_for = for_der(f, 0, h)
df_bac = bac_der(f, 0, h)
df_gra = gra_der(f, 0, h)

# print(timeit.timeit('for_der(f, x, h)', number = 1000, globals = globals()))
# print(timeit.timeit('bac_der(f, x, h)', number = 1000, globals = globals()))
# print(timeit.timeit('gra_der(f, x, h)', number = 1000, globals = globals()))

fig = go.Figure()
fig.add_scatter(x = N, y = df_for, mode = 'lines', name = 'Forward')
fig.add_scatter(x = N, y = df_bac, mode = 'lines', name = 'backward')
fig.add_scatter(x = N, y = df_gra, mode = 'lines', name = 'Centered')
fig.show()