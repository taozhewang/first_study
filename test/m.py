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
    x = step2_medium(x)
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