# %%
import numpy as np
import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# %% [markdown]
# We first define our Random walk which is mathematically given as
#
# $$
# \begin{equation}
# S_0=s,\quad S_t = X_t+S_{t-1}=S_0+\sum_{s=1}^t X_s
# \end{equation}
# $$
#
# Where 
#
# \begin{equation}
# X_t=
# \begin{cases}
# 1 &\text{If I get head for my coin toss}\\
# -1 &\text{If I get tail}
# \end{cases}
# \end{equation}
# 
# 
# 
# For a stock price, what you usually have is. Implement it
# 
# \begin{equation}
# S_t = S_{t-1} (1+u)^{X_t} = S_0 (1+u)^{\sum_{s=1}^t X_s}
# \end{equation}



#%%
# %%
shock = np.random.rand(40)
print(shock)
print(2 * np.round(shock) - 1)


# %%
def random_walk(startprice, days):
    price = np.zeros(days)
    shock = np.round(np.random.rand(days))
    price[0] = startprice
    for i in range(1, days):
        price[i] = price[i-1] + 2*shock[i]-1
    return price

# %% [markdown]
# This gives us the generation of sample random walks that we can print

# %%
print(random_walk(10, 20))

# %% [markdown]
# We want however to plot it

# %%
days = 10

# Define the X and Y axis

X = np.arange(days)
Y = random_walk(10, days)



# Define a figure object

fig = go.Figure()

fig.add_scatter(x = X, y = Y)
fig.layout.title = "My random walk"
fig.show()


# %%
# Plotting several scatters

days = 100000

X= np.arange(days)

#define an object

fig = go.Figure()

# add scatters to the figure

for i in np.arange(5):
    Y = random_walk(10, days)
    # append the new scatter with a name for each trace
    fig.add_scatter(x = X, y = Y, name = 'RW number %i'%i, line = {'width':1})
    

# add a title
fig.layout.title = "Sample paths of the random walks"

# display the random walks

fig.show()

# %% [markdown]
# # generate some histograms

# %%
X = np.random.randn(100000)


fig = go.Figure()
fig.add_histogram(x = X, histnorm = 'probability')
fig.show()

#%%

x = np.random.rand(10)
print(x)
x[x>0.6] = 1
print(x)
x[x<=0.6] = 0
print(x)