# import numpy as np
# import copy
# n = 3
# s = np.zeros((n, n), dtype = np.uint32)
# count =1
# success = []
# cache = []
# def fill(s, count, depth = 100):
#     blank = list(np.where(s == 0))
#     if depth == 0:
#         cache.append(s)
#         print(s,'s')
#         return False
#     if len(blank) == 0:
#         if check(s) and check(np.rot90(s)) and diagonal_check(s):
#             success.append(copy.deepcopy(s))
#         else:
#             print(s)
#             return 0
#     else:
#         for i in blank:
#             s[list(i)] = count
#             return fill(s, count + 1, depth - 1)
        
# def check(s):
#     for k in range(n):
#         if np.sum(s, axis = k) != n:
#             return False
#     return True
# def diagonal_check(s):
#     positive, negative = 0, 0
#     for k in range(n):
#         positive += s[k, k]
#         negative += s[k, n - k]
#     if positive == n and negative == n:
#         return True
#     else:
#         return False
    
# fill(s, count)
# print(success)

# # points24
# import numpy as np
# import copy
# import plotly.graph_objects as go

# # X = [3, 3, 8, 8]
# # p = []
# # # f(X, p)
# # # 8 / (3 - 8 / 3)
# # print(8 / (3 - 8 / 3))
# op=["+","-","*","/"]

# def eval(x,y,op_idx):
#     if op_idx == 0:
#         return x+y
#     elif op_idx == 1:
#         return x-y    
#     elif op_idx == 2:
#         return x*y
#     elif op_idx == 3:
#         return x/y
    
# def q(x, y):
#     return [x, y]
# def above(X):
#     # print(X)
#     return X[0]
# def below(X):
#     return X[1]
# def plus(X, Y):
#     x = above(X) * below(Y) + above(Y) * below(X)
#     y = below(X) * below(Y)
#     return q(x, y)
# def times(X, Y):
#     x = above(X) * above(Y)
#     y = below(X) * below(Y)
#     return q(x, y)
# def minus(X, Y):
#     x = above(X) * below(Y) - above(Y) * below(X)
#     y = below(X) * below(Y)
#     return q(x, y)
# def divide(X, Y):
#     x = above(X) * below(Y)
#     y = below(X) * above(Y)
#     return q(x, y)
# def equal(X, Y):
#     if above(X) * below(Y) == above(Y) * below(X):
#         return True
#     else:
#         return False
# def show(X):
#     return f'({above(X)} / {below(X)})'

# def f(X, p, target):
#     # print(X)
#     if len(X) == 1:
#         if equal(X[0], q(target, 1)):
#             # print(p)
#             return True
#         return False
#     else:
#         for i in range(len(X)):
#             for j in range(i + 1, len(X)):
#                 Y = copy.deepcopy(X)
#                 Y.pop(i)
#                 Y.pop(j - 1)
#                 # print(Y)
#                 # print(Y.append(plus(X[i], X[j])))
#                 # print(X[i], X[j])
#                 a = f(Y + [plus(X[i], X[j])], p + [f'{show(X[i])} + {show(X[j])} = {show(plus(X[i], X[j]))}'], target)
#                 b = f(Y + [times(X[i], X[j])], p + [f'{show(X[i])} * {show(X[j])} = {show(times(X[i], X[j]))}'], target)
#                 c = f(Y + [minus(X[i], X[j])], p + [f'{show(X[i])} - {show(X[j])} = {show(minus(X[i], X[j]))}'], target)
#                 d = f(Y + [minus(X[j], X[i])], p + [f'{show(X[j])} - {show(X[i])} = {show(minus(X[j], X[i]))}'], target)
#                 s = any([a, b, c, d])
#                 if not equal(X[j], q(0, 1)):
#                     a = f(Y + [divide(X[i], X[j])], p + [f'{show(X[i])} / {show(X[j])} = {show(divide(X[i], X[j]))}'], target)
#                     s = any([a, s])
#                 if not equal(X[i], q(0, 1)):
#                     a = f(Y + [divide(X[j], X[i])], p + [f'{show(X[j])} / {show(X[i])} = {show(divide(X[j], X[i]))}'], target)
#                     s = any([a, s])
#                 if s:
#                     return True
#                 else:
#                     return False

# # n1 = q(1, 1)
# # n2 = q(3, 1)
# # n3 = q(3, 1)
# # n4 = q(3, 1)
# # X = [n1, n2, n3, n4]
# # p = []
# # f(X, p, 24)

# def graph(x1, x2, target):
#     point_x1 = []
#     point_x2 = []
#     point_x3 = []
#     point_x4 = []
#     for y1 in range(x1, x2 + 1):
#         for y2 in range(y1, x2 + 1):
#             for y3 in range(y2, x2 + 1):
#                 for y4 in range(y3, x2 + 1):
#                     X = [q(y1, 1), q(y2, 1), q(y3, 1), q(y4, 1)]
#                     p = []
#                     s = f(X, p, target)
#                     if s:
#                         point_x1.append(y1)
#                         point_x2.append(y2)
#                         point_x3.append(y3)
#                         point_x4.append(y4)
#     return point_x1, point_x2, point_x3, point_x4

# # n = 10
# # point_x1, point_x2, point_x3, point_x4 = graph(1, n, 24)
# # fig = go.Figure()
# # point_x = (np.array(point_x1) - 1) * n + (np.array(point_x2) - 1)
# # point_y = (np.array(point_x3) - 1) * n + (np.array(point_x4) - 1)
# # fig.add_scatter(x = point_x, y = point_y, mode = 'markers', fillcolor = 'blue')
# # fig.show()

# n = 10
# fig = go.Figure()
# for i, target in enumerate(range(15, 31)):
    
#     point_x1, point_x2, point_x3, point_x4 = graph(1, n, target)
#     point_x = (np.array(point_x1) - 1) * n + (np.array(point_x2) - 1)   
#     point_y = (np.array(point_x3) - 1) * n + (np.array(point_x4) - 1)
#     fig.add_scatter(x = point_x, y = point_y, mode = 'markers', name = f'target = {target}, points = {len(point_x)}')
# fig.show()    

# # n = 10
# # target = 24
# # point_x1, point_x2, point_x3, point_x4 = graph(1, n, target)
# # fig = go.Figure()
# # t1 = np.array(point_x1)
# # t2 = np.array(point_x2)
# # t3 = np.array(point_x3)
# # t4 = np.array(point_x4)
# # while any(t1):
# #     i = t1[0]
# #     m1 = t1[t1 == i]
# #     m2 = t2[: len(m1)]
# #     m3 = t3[: len(m1)]
# #     m4 = t4[: len(m1)]
# #     fig.add_scatter3d(x = m2, y = m3, z = m4, mode = 'markers', name = f'x1 = {i}, points = {len(m1)}')
# #     t1 = t1[t1 != i]
# #     t2 = t2[len(m1) :]
# #     t3 = t3[len(m1) :]
# #     t4 = t4[len(m1) :]
# # fig.show()

# # land mine
# # land with mine is 1
# # land is checked is 2
# import numpy as np
# def map_origin(row, col, n, cor):
#     assert row > 0 and col > 0
#     M = np.zeros((row, col))
#     l = np.max([cor[1] - 1, 0])
#     r = np.min([cor[1] + 1, col - 1])
#     u = np.max([cor[0] - 1, 0])
#     d = np.min([cor[0] + 1, row - 1])
#     M[u : d + 1, l : r + 1] = 10
#     x, y = np.where(M == 0)
#     fill = np.random.choice(len(x), n, replace = False)

#     M[x[fill], y[fill]] = 1
#     M[u : d + 1, l : r + 1] = 0
#     return M

# # m = map_origin(10, 10, 30, [4, 5])
# # print(m)

# def map_show(M):
#     col = np.size(M, 1)
#     row = np.size(M, 0)
#     S = np.ones((row, col)) * 10
#     x, y = np.where(M == 2)
#     for i in range(len(x)):
#         l = np.max([y[i] - 1, 0])
#         r = np.min([y[i] + 1, col - 1])
#         u = np.max([x[i] - 1, 0])
#         d = np.min([x[i] + 1, row - 1])
#         t = np.where(M[u : d + 1, l : r + 1] == 1)
#         S[x[i], y[i]] = len(t[0])
#     for i in range(row):
#         for j in range(col):
#             if S[i, j] == 10:
#                 print('▢ ', end = '')
#             elif S[i, j] == 0:
#                 print('  ', end = '')
#             else:
#                 print(f'{int(S[i, j])} ', end = '')
#         print()
#     return


# m = np.array([[1, 0, 1, 1, 2, 2, 2, 1, 0, 0],
#                 [0, 0, 0, 1, 2, 2, 2, 0, 0, 0],
#                 [0, 0, 2, 0, 2, 2, 2, 1, 0, 1],
#                 [0, 1, 2, 1, 2, 2, 2, 1, 0, 0],
#                 [1, 1, 1, 1, 2, 2, 2, 0, 0, 1],
#                 [0, 1, 2, 2, 2, 2, 2, 1, 0, 0],
#                 [1, 2, 2, 2, 1, 1, 0, 1, 0, 1],
#                 [0, 2, 2, 2, 0, 0, 0, 0, 1, 0],
#                 [0, 2, 2, 2, 0, 0, 1, 1, 1, 1],
#                 [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]])
# map_show(m)

# import numpy as np
# # def idxyz(x, y, z):
# #     a = x + y + z
# #     b = x + y
# #     c = x
# #     return a * (a + 1) * (a + 2) / 6 + b * (b + 1) / 2 + c
# # for i in range(4):
# #     for j in range(4):
# #         for k in range(4):
# #             print(i, j, k, idxyz(i, j, k))

# def idxn(X):
#     n = len(X)
#     N = np.zeros(n)
#     Y = np.ones(n)
#     for i in range(n):
#         N[i] = sum(X[ : i + 1])
#         for j in range(i + 1):
#             Y[i] *= (N[i] + j) / (j + 1)
#     print(N, Y)
#     return sum(Y)

# for i in range(4):
#     for j in range(4):
#         for k in range(4):
#             for l in range(4):
#                 print(i, j, k, l, idxn([i, j, k, l]))

