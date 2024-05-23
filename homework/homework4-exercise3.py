import numpy as np
import plotly.graph_objects as go
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
#%%
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
#%%
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
#%%
#Q2

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

#%%
# Linear Regression Real Data
df = pd.read_csv('./homework/data/winequality.csv')

# print('head \n', df.head(), '\n')
# print('index \n', df.index, '\n')
# print('columns \n', df.columns, '\n')
# print('infos \n', df.info(), '\n')
# print('describe \n', df.describe(), '\n')

df = df.dropna().copy()


#%%
#Q1
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

# r_input = redtype[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
#                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
# r_input = sm.add_constant(r_input)
# model1 = sm.OLS(redtype['quality'], r_input)
# model1 = model1.fit()
# print(model1.summary())

# w_input = whitetype[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
#                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
# w_input = sm.add_constant(w_input)
# model2 = sm.OLS(whitetype['quality'], w_input)
# model2 = model2.fit()
# print(model2.summary())
#%%
#Q2
def find_feature():
    for t in [redtype, whitetype]:
        feature = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

        red_rsquare = []
        for redfeature in feature:
            redfeature = sm.add_constant(t[redfeature])
            model = sm.OLS(t['quality'], redfeature)
            model = model.fit()
            feature_rsquare = model.rsquared
            red_rsquare.append(feature_rsquare)
        largest_r_rsquare = np.max(red_rsquare)
        loc1 = red_rsquare.index(largest_r_rsquare)
        largest_r_feature = feature[loc1]
        feature.pop(loc1)

        red_rsquare = []
        for redfeature in feature:
            redfeature = sm.add_constant(t[[redfeature, largest_r_feature]])
            model = sm.OLS(t['quality'], redfeature)
            model = model.fit()
            feature_rsquare = model.rsquared
            red_rsquare.append(feature_rsquare)
        secondlargest_r_rsquare = np.max(red_rsquare)
        loc2 = red_rsquare.index(secondlargest_r_rsquare)
        secondlargest_r_feature = feature[loc2]

        print(f'the largest:{largest_r_feature},\nthe second largest:{secondlargest_r_feature}')
# find_feature()

#%%
#Clusters
#Q1

'''
N points in R^d are given as (x11, x12, ..., x1d), ..., (xn1, xn2, ..., xnd)
X = {X1, X2,..., Xn}
X = np.array([
    [x11, x12, x13, x14], 
    [x21, x22, x23, x24],
    [x31, x32, x33, x34]
])
'''
n = 20
d = 5
X = np.random.random_sample((n, d))

def aver(X):
    mu = np.sum(X, axis = 0) / n
    # print(mu)
    leftside = 0
    rightside = 0
    for i in range(n):
        for j in range(n):
            leftside += pow(np.linalg.norm(X[i] - X[j]), 2)
        rightside += pow(np.linalg.norm(X[i] - mu), 2)
    leftside = leftside / n
    rightside = rightside * 2

    print('leftside: ', leftside)
    print('rightside:', rightside)

# aver(X)
#%%
#Q2
k = 10
M = np.array([])
for ind in range(k):
    # pointnum = np.random.choice(np.arange(1, n))
    pointnum = k
    pointarray = np.random.choice(np.arange(1, n), pointnum, replace = False)
    X_k = X[pointarray, :]
    mu_k = np.mean(X_k, axis = 0)
    M = np.append(M, mu_k)
M = M.reshape((k, d))
# print('initialize:', M, '\n')
times = 0
def recur(X, M, times):
    
    family = {i: [] for i in range(np.size(M, axis = 0))}
    distances = 0
    for son, point in enumerate(X):
        belongs = []
        # print(point)
        for mumu in M:
            belong = np.linalg.norm(point - mumu)
            belongs.append(belong)
            # print(point, belong)
        minbelong = np.min(belongs)
        father = belongs.index(minbelong)
        distances += minbelong ** 2
        master = family[father]
        master.append(son)
        # print(father)
        family[father] = master
    mumufamily = np.array([])
    member = 0
    # print(family)
    for key in family:
        if family[key] != []: # maybe one of the mu's didn't get points
            newmumu = np.mean(X[family[key]], axis = 0)
            mumufamily = np.append(mumufamily, newmumu)
            member += 1
    # print(mumufamily)
    mumufamily = mumufamily.reshape(member, d)
    times += 1
    print(f'trial: {times}, distance: {distances}, \nmu: \n{mumufamily}')
    if np.array_equal(M, mumufamily):
        print('\n\ntotal trials:', times)
        # print(mumufamily)
        # print(family)
        for key in family:
            print(f'cluster{key}: \n{X[family[key]]}')
            print(f'mu: \n{mumufamily[key]}\n')
        return
    else:
        recur(X, mumufamily, times)
# recur(X, M, times)

#%%
#Q3
from sklearn.cluster import KMeans
import plotly.express as px
df = pd.read_csv('./homework/data/housing.csv')

print('head \n', df.head(), '\n')
print('index \n', df.index, '\n')
print('columns \n', df.columns, '\n')
print('infos \n', df.info(), '\n')
print('describe \n', df.describe(), '\n')

df = df.dropna().copy()

df = df.loc[:, ['longitude', 'latitude', 'median_income']]
print(df)

OMP_NUM_THREADS=1
N = 20
d = 5
# X = np.random.random_sample((n, d))
X = df
result = KMeans(n_clusters = 4).fit(X)
cluster = result.labels_
print(cluster)
df['color'] = cluster
print(df)
colors = {0: 'red', 1: 'yellow', 2: 'green', 3: 'blue'}
def color(C):
    A = np.array([])
    for n in C:
        A = np.append(A, colors[n])
    return A
fig = px.scatter(x = df['longitude'], y = df['latitude'], color = color(cluster))
fig.show()