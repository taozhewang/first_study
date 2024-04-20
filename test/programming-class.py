import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.options.plotting.backend = 'plotly'

s = pd.Series(['SH', 'BJ', 'London', np.nan, 'SG'], 
              index = ['a', 'b', 'c', 'd', 'e'])
print(s)
s = pd.Series(data = ['SH', 'BJ', 'London', np.nan, 'SG'], 
              index = ['a', 'b', 'c', 'd', 'e'], 
              name = 'My_serie',)
print(s)
