import numpy as np

a=np.ones((2,3))
b=np.ones((3,4))
c=np.matmul(a, b)
print(c.shape)  