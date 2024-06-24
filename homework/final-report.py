import numpy as np
import timeit
import numba as nb
from numba import jit
import copy
from scipy.optimize import root
# @jit
def random_choice_matrix(omega):
    rng = np.random.default_rng(seed = omega)
    print(rng)
    M = np.zeros((10, 10))
    for i in range(10):
        arr = np.arange(1, 11)
        a = rng.shuffle(arr)
        M[i, :] = arr
    return M
a = random_choice_matrix(1)
print(a)

def markov_game(omega):
    M = random_choice_matrix(omega)
    print(M)
    final = np.zeros(10)
    for start in range(10):
        i = 0
        j = start 
        while i < 9:
            
            j = (j + int(M[i, j])) % 10
            i += 1
        final[start] = M[i, j]    
    return final
final = markov_game(4)
print(final)
# Note: numbers in final are at most 3 kinds
# Note2: when numba is applied, its speed does not improve; also a lot of NumbaDeprecationWarning are given
print(timeit.timeit('markov_game(7)', number = 0, globals = globals()))

# Q2 : Performance
def mat_mult_slow(A, B, N, M, K):
    C = [[0 for _ in range(K)] for _ in range(N)]
    for i in range(N):
        for j in range(K):
            t = 0
            for k in range(M):
                t += A[i, k] * B[k, j]
            C[i][j] = t
    return C
def mat_mult_numpy(A, B):
    return A @ B

N = 1000
M = 10000
K = 5000
A = np.random.random((N, M))
B = np.random.random((M, K))

func1time = timeit.timeit('mat_mult_slow(A, B, N, M, K)', number = 0, globals = globals())
func2time = timeit.timeit('mat_mult_numpy(A, B)', number = 0, globals = globals())

print(f'mat_mult_slow: {func1time} \n mat_mult_numpy: {func2time}')

@nb.njit
def mat_mult_fast(A, B, N, M, K):
    C = [[0 for _ in range(K)] for _ in range(N)]
    for i in range(N):
        for j in range(K):
            t = 0
            for k in range(M):
                t += A[i, k] * B[k, j]
            C[i][j] = t
    return C
func3time = timeit.timeit('mat_mult_fast', number = 0, globals = globals())
print(f'mat_mult_fast: {func3time}')

# Q3: Pandas
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import plotly.graph_objects as go

breast = load_breast_cancer()
# print(breast)

data = breast['data']
target = breast['target']
feature_names = breast['feature_names']
# print(feature_names)
f = np.append(feature_names, 'label')
d = np.size(feature_names)
N = np.size(target)
target = np.reshape(target, (N, 1))
df = pd.DataFrame(columns = f, data = np.concatenate((data, target), axis = 1))
print(df)
dfd = pd.DataFrame(columns = feature_names, data = data)
dfdmean = dfd.mean()
dfdstd = dfd.std()
print(dfd)
print(dfdmean)
print(dfdstd)
for i, fn in enumerate(feature_names):
    dfd[fn] -= np.ones(np.size(dfd, axis = 0)) * dfdmean[i]
    dfd[fn] /= dfdstd[i]

dfret = dfd.dropna().copy()
Corr = dfret.corr()
eigenvalues, eigenvectors = np.linalg.eig(Corr)
print(eigenvalues, eigenvectors)
v1 = np.max(eigenvalues)
v1loc = np.where(eigenvalues == v1)[0]
y1 = eigenvectors[v1loc]
neigenvalues = np.delete(eigenvalues, v1loc)

v2 = np.max(neigenvalues)
v2loc = np.where(eigenvalues == v2)[0]
y2 = eigenvectors[v2loc]
# print(y1 - y2)
# print(np.size(y2))

pc1 = data @ y1.T
pc2 = data @ y2.T
# print(pc1 - pc2)
dfpca = pd.DataFrame(columns = ['pc1', 'pc2', 'label'], data = np.concatenate((pc1, pc2, target), axis = 1))

dfpcag = dfpca.groupby(['label'])


fig = go.Figure()
i = 0
for t in dfpcag:
    fig.add_scatter(x = np.array(t[1]['pc2']), y = np.array(t[1]['pc1']), mode = 'markers', name = f'label{i}')
    i = 1
fig.show()
# Explanation: using PCA to get 2 dim-d principle vectors y1 and y2: the influence of each feature_names
#              by plotting pc1 against pc2, the correlation between each feature_names can be seen

# Projects1: ODE
start = 0
end =10
timestep = [5e-01, 1e-01, 1e-02, 1e-03, 1e-04]
mu = 1000
y0 = 1
z0 = 0
def dy(z):
    return z
def dz(z, y):
    return mu * (1 - pow(y, 2)) * z - y
def explicit(z0, y0, start, end, timestep, dy, dz):
    T = np.arange(start, end, timestep)
    Y = np.arange(start, end, timestep)
    Z = np.arange(start, end, timestep)
    Y[start] = y0
    Z[start] = z0
    # print((end - start) / timestep)
    for i in range(int((end - start) / timestep - 1)):
        Y[i + 1] = Y[i] + timestep * dy(Z[i])
        Z[i + 1] = Z[i] + timestep * dz(Z[i], Y[i])
    return Y
def RK(z0, y0, start, end, timestep, dy, dz):
    T = np.arange(start, end, timestep)
    Y = np.arange(start, end, timestep)
    Z = np.arange(start, end, timestep)
    Y[start] = y0
    Z[start] = z0
    for i in range(int((end - start) / timestep - 1)):
        y1 = dy(Z[i])
        z1 = dz(Z[i], Y[i])
        y2 = dy(Z[i] + timestep * z1 / 2)
        z2 = dz(Z[i] + timestep * z1 / 2, Y[i] + timestep * y1 / 2)
        y3 = dy(Z[i] + timestep * z2 / 2)
        z3 = dz(Z[i] + timestep * z2 / 2, Y[i] + timestep * y2 / 2)
        y4 = dy(Z[i] + timestep * z3)
        z4 = dz(Z[i] + timestep * z3, Y[i] + timestep * y3)
        Y[i + 1] = Y[i] + timestep * (y1 + 2 * y2 + 2 * y3 + y4) / 6
        Z[i + 1] = Z[i] + timestep * (z1 + 2 * z2 + 2 * z3 + z4) / 6
    return Y

def implicit(z0, y0, start, end, timestep):
    T = np.arange(start, end, timestep)
    Y = np.arange(start, end, timestep)
    Z = np.arange(start, end, timestep)
    Y[start] = y0
    Z[start] = z0
    for i in range(int((end - start) / timestep - 1)):
        def fy(y, z):
            return Y[i] + timestep * z - y
        def fz(y, z):
            return Z[i] + timestep * (mu * (1- pow(y, 2)) * z - y) - z
        def func(variables):
            y, z = variables
            return [fy(y, z), fz(y, z)]
        result = root(func, [Y[i], Z[i]])
        Y[i + 1], Z[i + 1] = result.x
    return Y

for t in timestep:
    Y1 = explicit(z0, y0, start, end, t, dy, dz)
    Y2 = RK(z0, y0, start, end, t, dy, dz)    
    Y3 = implicit(z0, y0, start, end, t)
    fig = go.Figure()
    fig.add_scatter(x = np.arange(start, end, t), y = Y1, mode = 'lines+markers', name = f'explicit with timestep {t}')
    fig.add_scatter(x = np.arange(start, end, t), y = Y2, mode = 'lines+markers', name = f'RK with timestep {t}')
    fig.add_scatter(x = np.arange(start, end, t), y = Y3, mode = 'lines+markers', name = f'implicit with timestep {t}')
    fig.show()
t = 1e-04
t1 = timeit.timeit('explicit(z0, y0, start, end, t, dy, dz)', number = 1, globals = globals())
t2 = timeit.timeit('RK(z0, y0, start, end, t, dy, dz)', number = 1, globals = globals())
t3 = timeit.timeit('implicit(z0, y0, start, end, t)', number = 1, globals = globals())
print(f'explicit: {t1} \n RK: {t2} \n implicit: {t3}')

@jit
def fastexplicit(z0, y0, start, end, timestep, dy, dz):
    
    Y = []
    Z = []
    Y.append(y0)
    Z.append(z0)
    # print((end - start) / timestep)
    for i in range(int((end - start) / timestep - 1)):
        y = Y[i] + timestep * dy(Z[i])
        z = Z[i] + timestep * dz(Z[i], Y[i])
        Y.append(y)
        Z.append(z)
    return Y

@jit
def fastRK(z0, y0, start, end, timestep, dy, dz):
    Y = []
    Z = []
    Y.append(y0)
    Z.append(z0)
    for i in range(int((end - start) / timestep - 1)):
        y1 = dy(Z[i])
        z1 = dz(Z[i], Y[i])
        y2 = dy(Z[i] + timestep * z1 / 2)
        z2 = dz(Z[i] + timestep * z1 / 2, Y[i] + timestep * y1 / 2)
        y3 = dy(Z[i] + timestep * z2 / 2)
        z3 = dz(Z[i] + timestep * z2 / 2, Y[i] + timestep * y2 / 2)
        y4 = dy(Z[i] + timestep * z3)
        z4 = dz(Z[i] + timestep * z3, Y[i] + timestep * y3)
        y = Y[i] + timestep * (y1 + 2 * y2 + 2 * y3 + y4) / 6
        z = Z[i] + timestep * (z1 + 2 * z2 + 2 * z3 + z4) / 6
        Y.append(y)
        Z.append(z)
    return Y

# @jit
def fastimplicit(z0, y0, start, end, timestep):
    i = 0
    Y = []
    Z = []
    Y.append(y0)
    Z.append(z0)
    # def func(variables):
    #     y, z = variables    
    #     eq1 = Y[i] + timestep * z - y
    #     eq2 = Z[i] + timestep * (mu * (1- pow(y, 2)) * z - y) - z
    #     return [eq1, eq2]
    for i in range(int((end - start) / timestep - 1)):
        h = 1e-04
        yk, zk = Y[i], Z[i]
        result = root(lambda v: [yk + h * v[1] - v[0], zk + h * (mu * (1- pow(v[0], 2)) * v[1] - v[0]) - v[1]], [yk, zk])
        y, z = result.x
        Y.append(y)
        Z.append(z)
    return Y

t = 1e-04
t1 = timeit.timeit('fastexplicit(z0, y0, start, end, t, dy, dz)', number = 1, globals = globals())
t2 = timeit.timeit('fastRK(z0, y0, start, end, t, dy, dz)', number = 1, globals = globals())
t3 = timeit.timeit('fastimplicit(z0, y0, start, end, t)', number = 1, globals = globals())
print(f'fastexplicit: {t1} \n fastRK: {t2} \n fastimplicit: {t3}')
# Note: euler implicit scheme seems hard to work by numba
# 'Cannot capture the non-constant value associated with variable '**' in a function that may escape.' appear all the time
