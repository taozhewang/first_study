import numpy as np
import copy
Z_h = np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])
Z_t = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0]])

Y_h = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])
Y_t = np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0]])

X_h = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])
X_t = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0]])

# def generate_():
#     u = np.zeros((24, 24), dtype = np.uint32)
#     for i in np.array([[1, 0], [2, 1], [3, 2], [0, 3]]):
#         x, y = i * 3
#         u[x : x + 3, y : y + 3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
#     for i in np.array([[4, 4], [5, 5], [6, 6], [7, 7]]):
#         x, y = i * 3
#         u[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     d = np.zeros((24, 24), dtype = np.uint32)
#     for i in np.array([[5, 4], [6, 5], [7, 6], [4, 7]]):
#         x, y = i * 3
#         d[x : x + 3, y : y + 3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
#     for i in np.array([[0, 0], [1, 1], [2, 2], [3, 3]]): 
#         x, y = i * 3
#         d[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     f = np.zeros((24, 24), dtype = np.uint32)
#     for i in np.array([[4, 0], [0, 1], [5, 4], [1, 5]]):
#         x, y = i * 3
#         f[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
#     for i in np.array([[2, 2], [3, 3], [6, 6], [7, 7]]):
#         x, y = i * 3
#         f[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     b = np.zeros((24, 24), dtype = np.uint32)
#     for i in np.array([[3, 2], [7, 3], [2, 6], [6, 7]]):
#         x, y = i * 3
#         b[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
#     for i in np.array([[0, 0], [1, 1], [4, 4], [5, 5]]): 
#         x, y = i * 3
#         b[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     r = np.zeros((24, 24), dtype = np.uint32)
#     for i in np.array([[5, 1], [1, 2], [6, 5], [2, 6]]):
#         x, y = i * 3
#         r[x : x + 3, y : y + 3] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
#     for i in np.array([[0, 0], [3, 3], [4, 4], [7, 7]]):
#         x, y = i * 3
#         r[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     l = np.zeros((24, 24), dtype = np.uint32)
#     for i in np.array([[4, 0], [0, 3], [7, 4], [3, 7]]):
#         x, y = i * 3
#         l[x : x + 3, y : y + 3] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
#     for i in np.array([[1, 1], [2, 2], [5, 5], [6, 6]]): 
#         x, y = i * 3
#         l[x : x + 3, y : y + 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     return u, d, f, b, r, l
# u, d, f, b, r, l = generate_()
# # print(u, d, f, b, r, l, sep = '\n')
# def act(x, n):
#     print(n)
#     return n @ x
# def lurd(x):
#     act(act(act(act(x, u.T), r.T), u), r)
def generate(left, x, n):
    l = np.where(np.diag(left) == 0)
    cl = np.where(x[l] == 1)
    blb = (cl[1] // 3) * 3
    r = np.identity(24, dtype = np.uint32)
    if n == 'z':
        for i in blb:
            r[i : i + 3, i : i + 3] = np.array([[0, 1, 0],
                                                [1, 0, 0],
                                                [0, 0, 1]])
    elif n == 'y':
        for i in blb:
            r[i : i + 3, i : i + 3] = np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
    else:
        for i in blb:
            r[i : i + 3, i : i + 3] = np.array([[1, 0, 0],
                                                [0, 0, 1],
                                                [0, 1, 0]])
    return r

def act(x, n):
    if n == 'u':
        R_z = generate(Z_h, x, 'z')
        return Z_h @ x @ R_z
    elif n == 'ut':
        R_z = generate(Z_h, x, 'z')
        return Z_h.T @ x @ R_z
    elif n == 'd':
        R_z = generate(Z_t, x, 'z')
        return Z_t @ x @ R_z
    elif n == 'dt':
        R_z = generate(Z_t, x, 'z')
        return Z_t.T @ x @ R_z
    elif n == 'f':
        R_x = generate(X_h, x, 'x')
        return X_h @ x @ R_x
    elif n == 'ft':
        R_x = generate(X_h, x, 'x')
        return X_h.T @ x @ R_x
    elif n == 'b':
        R_x = generate(X_t, x, 'x')
        return X_t @ x @ R_x
    elif n == 'bt':
        R_x = generate(X_t, x, 'x')
        return X_t.T @ x @ R_x
    elif n == 'r':
        R_y = generate(Y_h, x, 'y')
        return Y_h @ x @ R_y
    elif n == 'rt':
        R_y = generate(Y_h, x, 'y')
        return Y_h.T @ x @ R_y
    elif n == 'l':
        R_y = generate(Y_t, x, 'y')
        return Y_t @ x @ R_y
    elif n == 'lt':
        R_y = generate(Y_t, x, 'y')
        return Y_t.T @ x @ R_y
    else:
        return x
def lurd(x):
    return act(act(act(act(x, 'ut'), 'rt'), 'u'), 'r')

cube = np.zeros((8, 24), dtype = np.uint32)
for i in range(8):
    cube[i, 3 * i] = 1
def Rubik_cube_simulation(cube):
    rcube = copy.deepcopy(cube)
    for _ in range(30):
        i = np.random.choice(['u', 'd', 'f', 'b', 'l', 'r'])
        rcube = act(rcube, i)
    return rcube
ex = Rubik_cube_simulation(cube)
print(ex)

# cube = np.zeros((24, 8), dtype = np.uint32)
# for i in range(8):
#     cube[3 * i, i] = 1
# def Rubik_cube_simulation(cube):
#     direction = [u, d, f, b, l, r]
#     rcube = copy.deepcopy(cube)
#     for _ in range(30):
#         i = np.random.choice([0, 1, 2, 3, 4, 5])
#         rcube = act(rcube, direction[i])
#     return rcube
# ex = Rubik_cube_simulation(cube)
# print(ex)

def step1(x):
    r, c = np.where(x[:, 12 : 15] == 1)
    if r == [0]:
        if c == [0]:
            return act(x, 'l')
        elif c == [1]:
            return act(x, 'f')
        elif c == [2]:
            return act(act(act(x, 'f'), 'f'), 'dt')
    elif r == [1]:
        if c == [0]:
            return act(act(x, 'r'), 'dt')
        elif c == [1]:
            return act(act(x, 'ft'), 'dt')
        elif c == [2]:
            return act(act(x, 'f'), 'f')
    elif r == [2]:
        if c == [0]:
            return act(act(act(x, 'u'), 'u'), 'l')
        elif c == [1]:
            return act(act(act(x, 'u'), 'u'), 'f')
        elif c == [2]:
            return act(act(act(x, 'r'), 'r'), 'dt')
    elif r == [3]:
        if c == [0]:
            return act(act(x, 'lt'), 'd')
        elif c == [1]:
            return act(act(x, 'b'), 'd')
        elif c == [2]:
            return act(act(x, 'l'), 'l')
    elif r == [4]:
        if c == [0]:
            return act(act(x, 'l'), 'd')
        elif c == [1]:
            return act(act(x, 'f'), 'dt')
        elif c == [2]:
            return x
    elif r == [5]:
        if c == [0]:
            return act(act(act(x, 'r'), 'd'), 'd')
        elif c == [1]:
            return act(x, 'f')
        elif c == [2]:
            return act(x, 'dt')
    elif r == [6]:
        if c == [0]:
            return act(act(x, 'rt'), 'dt')
        elif c == [1]:
            return act(act(x, 'bt'), 'd')
        elif c == [2]:
            return act(act(x, 'd'), 'd')
    elif r == [7]:
        if c == [0]:
            return act(x, 'lt')
        elif c == [1]:
            return act(act(act(x, 'b'), 'd'), 'd')
        elif c == [2]:
            return act(x, 'd')
        
# def step1(x):
#     r = np.where(x[:, 4] == 1)
#     r, c = r[0] // 3, r[0] % 3
#     if r == [0]:
#         if c == [0]:
#             return act(x, l)
#         elif c == [1]:
#             return act(x, f)
#         elif c == [2]:
#             return act(act(act(x, f), f), d.T)
#     elif r == [1]:
#         if c == [0]:
#             return act(act(x, r), d.T)
#         elif c == [1]:
#             return act(act(x, f.T), d.T)
#         elif c == [2]:
#             return act(act(x, f.T), f.T)
#     elif r == [2]:
#         if c == [0]:
#             return act(act(act(x, u), u), l)
#         elif c == [1]:
#             return act(act(act(x, u), u), f)
#         elif c == [2]:
#             return act(act(act(x, r), r), d.T)
#     elif r == [3]:
#         if c == [0]:
#             return act(act(x, l.T), d)
#         elif c == [1]:
#             return act(act(x, b), d)
#         elif c == [2]:
#             return act(act(x, l), l)
#     elif r == [4]:
#         if c == [0]:
#             return act(act(x, l), d)
#         elif c == [1]:
#             return act(act(x, f), d.T)
#         elif c == [2]:
#             return x
#     elif r == [5]:
#         if c == [0]:
#             return act(act(act(x, r), d), d)
#         elif c == [1]:
#             return act(x, f)
#         elif c == [2]:
#             return act(x, d.T)
#     elif r == [6]:
#         if c == [0]:
#             return act(act(x, r.T), d.T)
#         elif c == [1]:
#             return act(act(x, b.T), d)
#         elif c == [2]:
#             return act(act(x, d), d)
#     elif r == [7]:
#         if c == [0]:
#             return act(x, l.T)
#         elif c == [1]:
#             return act(act(act(x, b), d), d)
#         elif c == [2]:
#             return act(x, d)

# def step2_medium(x):
#     r = np.where(x[:, 5] == 1)
#     print(r)
#     r = r[0] // 3, r[0] % 3
#     if r == [0]:
#         return act(x, u)
#     elif r == [1]:
#         return x
#     elif r == [2]:
#         return act(x, u.T)
#     elif r == [3]:
#         return act(act(x, u), u)
#     elif r == [5]:
#         return x
#     elif r == [6]:
#         return act(x, r.T)
#     elif r == [7]:
#         return act(act(x, b), r.T)
   
# def step(x, i):
#     count = 0
#     while True:
#         print(x)
#         r = np.where(x[:, i] == 1)
#         r, c = r[0] // 3, r[0] % 3
#         if r == [5] and c == [2]:
#             return x
#         elif count > 5:
#             return False
#         else:
#             count += 1
#             x = lurd(x)

# def step3_medium(x):
#     x = act(x, d.T)
#     r = np.where(x[:, 6] == 1)
#     r = r[0] // 3
#     if r == [0]:
#         return act(x, u)
#     elif r == [1]:
#         return x
#     elif r == [2]:
#         return act(x, u.T)
#     elif r == [3]:
#         return act(act(x, u), u)
#     elif r == [5]:
#         return x
#     elif r == [6]:
#         return act(x, r.T)

# def step4_medium(x):
#     x = act(x, d.T)
#     r = np.where(x[:, 7] == 1)
#     r = r[0] // 3, r[0] % 3
#     if r == [0]:
#         return act(x, u)
#     elif r == [1]:
#         return x
#     elif r == [2]:
#         return act(x, u.T)
#     elif r == [3]:
#         return act(act(x, u), u)
#     elif r == [5]:
#         return x

# def step5_medium(x):
#     x = act(act(act(act(x, l), l), r), r)
#     r = np.where(x[:, 0] == 1)
#     r = r[0] // 3, r[0] % 3
#     if r == [4]:
#         return act(x, d)
#     elif r == [5]:
#         return x
#     elif r == [6]:
#         return act(x, d.T)
#     elif r == [7]:
#         return act(act(x, d), d)

# def step2(x):
#     x = act(x, d.T)
#     cl = np.where(x[15 : 18, :] == 1)
#     cl = (cl[0] // 3) * 3
#     count = 0
#     while True:
#         r = np.where(x[:, cl[0]] == 1)
#         r, c = r // 3, r % 3
#         if r == [5] and c == [2]:
#             return x
#         elif count > 5:
#             return False
#         else:
#             count += 1
#             x = lurd(x)

# def adjust(x):
#     for i in range(3):
#         for j in range(3):
#             x = lurd(x)
#         x = act(x, d.T)
#     x = act(x, d.T)
#     for _ in range(3):
#         x = lurd(x)
#     return x
# def step9_medium(x):
#     r, c = np.where(x[:, 4])
#     r, c = r // 3, r % 3
#     if r == [0]:
#         return act(act(x, u), u)
#     elif r == [1]:
#         return act(x, u)
#     elif r == [2]:
#         return x
#     else:
#         return act(x, u.T)
# def step9(x):
#     block = np.where(x[15 : 18, :] == 1)
#     bl = block[0] // 3
#     blockk = np.where(x[12 : 15, :] == 1)
#     bll = blockk[0] // 3
#     if bl == [1]:
#         if bll == [2]:
#             return x
#         else:
#             return adjust(x)
#     elif bl == [2]:
#         if bll == [1]:
#             x = act(act(x, u.T), d.T)
#             return adjust(x)
#         else:
#             x = act(act(x, u), d)
#             return adjust(x)
#     elif bl == [3]:
#         if bll == [1]:
#             x = act(act(act(act(x, u), u), d), d)
#             return adjust(x)
#         else:
#             x = adjust(x)
#             x = act(act(x, u), d)
#             return adjust(x)
# def step10_medium(x):
#     r1, c1 = np.where(x[:, 4])
#     r1, c1 = r1 // 3, r1 % 3
#     if r1 == [0]:
#         x = act(x, u)
#     elif r1 == [2]:
#         x = act(x, u.T)
#     elif r1 == [3]:
#         x = act(act(x, u), u)
#     r2, c2 = np.where(x[:, 0])
#     r2, c2 = r2 // 3, r2 % 3
#     if r2 == [4]:
#         return act(x, d)
#     elif r2 == [6]:
#         return act(x, d.T)
#     elif r2 == [7]:
#         return act(act(x, d), d)
#     else:
#         return x
# def step10(x):
#     return act(act(act(act(x, f), f), b), b)

def step2_medium(x):
    r, c = np.where(x[:, 15 : 18] == 1)
    if r == [0]:
        return act(x, 'u')
    elif r == [1]:
        return x
    elif r == [2]:
        return act(x, 'ut')
    elif r == [3]:
        return act(act(x, 'u'), 'u')
    elif r == [5]:
        return x
    elif r == [6]:
        return act(x, 'rt')
    elif r == [7]:
        return act(act(x, 'b'), 'rt')

def step(x, i):
    count = 0
    while True:
        r, c = np.where(x[:, i * 3 : (i + 1) * 3] == 1)
        if r == [5] and c == [2]:
            return x
        elif count > 5:
            return False
        else:
            count += 1
            x = lurd(x)

def step3_medium(x):
    x = act(x, 'dt')
    r, c = np.where(x[:, 18 : 21] == 1)
    if r == [0]:
        return act(x, 'u')
    elif r == [1]:
        return x
    elif r == [2]:
        return act(x, 'ut')
    elif r == [3]:
        return act(act(x, 'u'), 'u')
    elif r == [5]:
        return x
    elif r == [6]:
        return act(x, 'rt')

def step4_medium(x):
    x = act(x, 'dt')
    r, c = np.where(x[:, 21 : 24] == 1)
    if r == [0]:
        return act(x, 'u')
    elif r == [1]:
        return x
    elif r == [2]:
        return act(x, 'ut')
    elif r == [3]:
        return act(act(x, 'u'), 'u')
    elif r == [5]:
        return x

def step5_medium(x):
    x = act(act(act(act(x, 'l'), 'l'), 'r'), 'r')
    r, c = np.where(x[:, 0 : 3] == 1)
    if r == [4]:
        return act(x, 'd')
    elif r == [5]:
        return x
    elif r == [6]:
        return act(x, 'dt')
    elif r == [7]:
        return act(act(x, 'd'), 'd')

def step2(x):
    x = act(x, 'dt')
    cl = np.where(x[5, :] == 1)
    cl = (cl[0] // 3) * 3
    count = 0
    while True:
        r, c = np.where(x[:, cl[0] : cl[0] + 3] == 1)
        if r == [5] and c == [2]:
            return x
        elif count > 5:
            return False
        else:
            count += 1
            x = lurd(x)

def adjust(x):
    for i in range(3):
        for j in range(3):
            x = lurd(x)
        x = act(x, 'dt')
    x = act(x, 'dt')
    for _ in range(3):
        x = lurd(x)
    return x
def step9_medium(x):
    r, c = np.where(x[:, 12 : 15])
    if r == [0]:
        return act(act(x, 'u'), 'u')
    elif r == [1]:
        return act(x, 'u')
    elif r == [2]:
        return x
    else:
        return act(x, 'ut')
def step9(x):
    block = np.where(x[5, :] == 1)
    bl = block[0] // 3
    blockk = np.where(x[4, :] == 1)
    bll = blockk[0] // 3
    if bl == [1]:
        if bll == [2]:
            return x
        else:
            return adjust(x)
    elif bl == [2]:
        if bll == [1]:
            x = act(act(x, 'ut'), 'dt')
            return adjust(x)
        else:
            x = act(act(x, 'u'), 'd')
            return adjust(x)
    elif bl == [3]:
        if bll == [1]:
            x = act(act(act(act(x, 'u'), 'u'), 'd'), 'd')
            return adjust(x)
        else:
            x = adjust(x)
            x = act(act(x, 'u'), 'd')
            return adjust(x)
def step10_medium(x):
    r1, c1 = np.where(x[:, 12 : 15])
    if r1 == [0]:
        x = act(x, 'u')
    elif r1 == [2]:
        x = act(x, 'ut')
    elif r1 == [3]:
        x = act(act(x, 'u'), 'u')
    r2, c2 = np.where(x[:, 0 : 3])
    if r2 == [4]:
        return act(x, 'd')
    elif r2 == [6]:
        return act(x, 'dt')
    elif r2 == [7]:
        return act(act(x, 'd'), 'd')
    else:
        return x
def step10(x):
    return act(act(act(act(x, 'f'), 'f'), 'b'), 'b')

def solve(x):
    x = step1(x)
    print(x)
    x = step2_medium(x)
    print(x)
    x = step(x, 5)
    x = step3_medium(x)
    x = step(x, 6)
    x = step4_medium(x)
    x = step(x, 7)
    x = step5_medium(x)
    x = step(x, 0)
    for _ in range(3):
        x = step2(x)
    x = step9_medium(x)
    x = step9(x)
    x = step10_medium(x)
    x = step10(x)
    return x

result = solve(ex)
print(result)

# # random playing
# def reversion_success(x):
#     initial = np.zeros((8, 24), dtype = np.uint32)
#     for i in range(8):
#         initial[i, 3 * i + 2] = 1
#     v = copy.deepcopy(initial)
#     another_version = []
#     unwise_action = ['r', 'r', 'f', 'u', 'u']
#     it_would_be_wise_my_friend =['l', 'l', 'b', 'd', 'd']
#     by_the_way = ['u', 'f', 'u', 'r', 'f', 'r']
#     also = ['d', 'b', 'd', 'l', 'b', 'l']
#     for orientation in range(5):
#         another_version.append(v)
#         for i in range(3):
#             v = act(v, by_the_way[i])
#             v = act(v, also[i])
#             another_version.append(v)
#         v = act(v, unwise_action[i])
#         v = act(v, it_would_be_wise_my_friend[i])
#     for i in another_version:
#         if np.array_equal(x, i):
#             return True
#     return False

# def random_act(x):
#     action = np.random.choice(['u', 'd', 'f', 'b', 'l', 'r', 'ut', 'dt', 'ft', 'bt', 'lt', 'rt'])
#     return act(x, action), action

# def random_move(x):
#     upper_bound = 10
#     count = 0
#     action_list = []
#     while count < upper_bound:
#         x, action = random_act(x)
#         action_list.append(action)
#         count += 1
#     return x, action_list
# print(reversion_success(ex))
# print(random_move(ex))