import numpy as np
import plotly.graph_objects as go
import pandas as pd
import statsmodels

# Regression
b = np.array([1, 1, -0.5]).T
mu = np.array([1, 0]).T
sigma1 = 1
sigma2 = 0.5
rho = 0.4
Sigma = np.array(
    [
        [sigma1 ** 2, sigma1 * sigma2 * rho],
        [sigma1 * sigma2 * rho, sigma2 ** 2]
    ]
)

sigma = 2

# We prepare the dataset
# Fix random number generator and number of samples
rng = np.random.default_rng(seed = 150)
N = 1000
# generate samples
X = rng.multivariate_normal(mu, Sigma, size = N)        # size N x 2
epsilon = rng.normal(0, sigma, size = N)                # size N
# Add an axis of 1 to X
X = np.append(np.ones((N, 1)), X, axis = 1)             # append Nx1 to Nx2 -> Nx3
# Generate Y
Y = X.dot(b) + epsilon                                  # size N
# print(X)
# print(Y)
#Q1
def plotyx():
    fig = go.Figure()
    fig.add_scatter(x = X[:, 1], y = Y, mode = 'markers', name = 'Y and X_1')
    fig.add_scatter(x = X[:, 2], y = Y, mode = 'markers', name = 'Y and X_2')
    fig.show()
# plotyx()
def plotxx():
    fig = go.Figure()
    fig.add_scatter(x = X[:, 1], y = X[:, 2], mode = 'markers', name = 'X_1 and X_2')
    fig.show()
# plotxx()
def bn_and_err():
    inver = np.linalg.inv((X.T).dot(X))
    matrixx = inver.dot(X.T)
    com_b = matrixx.dot(Y)
    print('computation B:', com_b)
    print('original B:', b)
    err = Y - X.dot(com_b.T)
    print(err)

    fig = go.Figure()
    fig.add_bar(x = np.arange(len(err)), y = err)
    fig.show()
    return
# bn_and_err()

#Q2
# def f(N, M):

def trial(N):
    b = np.array([1, 1, -0.5]).T
    mu = np.array([1, 0]).T
    sigma1 = 1
    sigma2 = 0.5
    rho = 0.4
    Sigma = np.array(
        [
            [sigma1 ** 2, sigma1 * sigma2 * rho],
            [sigma1 * sigma2 * rho, sigma2 ** 2]
        ]
    )
    sigma = 2
    rng = np.random.default_rng(seed = None)
    X = rng.multivariate_normal(mu, Sigma, size = N)        
    epsilon = rng.normal(0, sigma, size = N)                
    X = np.append(np.ones((N, 1)), X, axis = 1)             
    Y = X.dot(b) + epsilon   

    inver = np.linalg.inv((X.T).dot(X))
    matrixx = inver.dot(X.T)
    com_b = matrixx.dot(Y)
    err = Y - X.dot(com_b.T)
    return com_b, err
def f(N, M):
    B, E = trial(N)
    for _ in range(M - 1):
        b_M, err_M = trial(N)
        B = np.concatenate((B, b_M), axis = 0)
        E = np.concatenate((E, err_M), axis = 0)
    B = np.reshape(B, (M, 3)).T
    E = np.reshape(E, (M, N)).T

    fig = go.Figure()
    fig.add_bar(x = np.arange(M), y = B[0, :], name = 'a')
    fig.add_bar(x = np.arange(M), y = B[1, :], name = 'b1')
    fig.add_bar(x = np.arange(M), y = B[2, :], name = 'b2')
    fig.show()

    fig = go.Figure()
    fig.add_bar(x = np.arange(M), y = np.mean(E, axis = 0), text = 'mean')
    fig.show()
    fig = go.Figure()
    fig.add_bar(x = np.arange(M), y = np.std(E, axis = 0), text = 'standard diviation')
    fig.show()
# f(10, 100)
# f(100, 100)
# f(1000, 100)

# Linear Regression Real Data
df = pd.read_csv('./homework/data/winequality.csv')

# print('head \n', df.head(), '\n')
# print('index \n', df.index, '\n')
# print('columns \n', df.columns, '\n')
# print('infos \n', df.info(), '\n')
# print('describe \n', df.describe(), '\n')


redtype = df[df['type'] == 'red']
whitetype = df[df['type'] == 'white']

# print(redtype)
# print(whitetype)

# print(np.min(df['quality']))
# print(np.max(df['quality']))
def plot_relations():
    for t in [redtype, whitetype]:
        for n in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']:
            fig = go.Figure()
            for i in np.sort(df['quality'].unique()):
                curr_ = t[t['quality'] == i]
                kinds = np.sort(curr_[n].unique())
                y = []
                for j in kinds:
                    _kind = curr_[curr_[n] == j][n]
                    kind_number = len(_kind)
                    y.append(kind_number)
                fig.add_scatter(x = kinds, y = y, mode = 'lines+markers', name = f'{n} -> quality{i}')
                fig.update_layout(yaxis_title = 'number', xaxis_title = f'{n}')
            fig.show()
# plot_relations()



