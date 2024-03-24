import numpy as np
import copy
n = 3
s = np.zeros((n, n), dtype = np.uint32)
count =1
success = []
cache = []
def fill(s, count, depth = 100):
    blank = list(np.where(s == 0))
    if depth == 0:
        cache.append(s)
        print(s,'s')
        return False
    if len(blank) == 0:
        if check(s) and check(np.rot90(s)) and diagonal_check(s):
            success.append(copy.deepcopy(s))
        else:
            print(s)
            return 0
    else:
        for i in blank:
            s[list(i)] = count
            return fill(s, count + 1, depth - 1)
        
def check(s):
    for k in range(n):
        if np.sum(s, axis = k) != n:
            return False
    return True
def diagonal_check(s):
    positive, negative = 0, 0
    for k in range(n):
        positive += s[k, k]
        negative += s[k, n - k]
    if positive == n and negative == n:
        return True
    else:
        return False
    
fill(s, count)
print(success)