#%%
import plotly.graph_objects as go
import pandas as pd
import numpy as np
df = pd.read_csv('./test/data/CSI.csv')
# print('head', df.head())
# print('index', df.index)
# print('columns', df.columns)
# print('infos', df.info())
# print('describe', df.describe())

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df)
dfret = df.pct_change().dropna().copy()
mu = dfret.mean()
print(mu)
Sigma = dfret.cov()

eigenvalues, eigenvectors = np.linalg.eig(Sigma)
print(eigenvalues, eigenvectors)

idx = eigenvalues.argsort()[::-1]
order_eigenvalues = eigenvalues[idx]
print(idx)
cumev = order_eigenvalues.cumsum() / order_eigenvalues.sum()

# fig = go.Figure()
# fig.add_scatter(x = np.arange(1, len(order_eigenvalues)), y = cumev, mode = 'lines+markers')
# fig.show()

gap = np.percentile(order_eigenvalues, 50)
#%%
