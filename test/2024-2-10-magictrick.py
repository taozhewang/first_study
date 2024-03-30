# # first version
# length = 13
# a = []
# for i in range(length):
#     a.append(i+1)
# count = 1
# while len(a) > 1:
#     if count == 1:
#         t = a[0]
#         for j in range(len(a) - 1):
#             a[j] = a[j + 1]
#         a[len(a) - 1] = t
#         count = 0
#     if count == 0:
#         a.pop(0)
#         count = 1
# print(a[0])

# # second version
# import numpy as np
# length = 7
# a = np.zeros((length, 1), dtype = np.uint32)
# for i in range(length):
#     a[i, 0] = i + 1

# def permutation_matrix(a):
#     p = np.zeros((len(a), len(a)), dtype = np.uint32)
#     for i in range(len(a) - 1):
#         p[i, i + 1] = 1
#     p[len(a) - 1, 0] = 1
#     return(p)

# def cut_permutation_matrix(a):
#     q = np.zeros((len(a) - 1, len(a)), dtype = np.uint32)
#     for i in range(len(a) - 1):
#         q[i, i + 1] = 1
#     return(q)

# def leave_behind(a):
#     if len(a) == 1:
#         print(a[0, 0])
#     else:
#         p = permutation_matrix(a)
#         return(throw_away(np.matmul(p, a)))

# def throw_away(a):
#     if len(a) == 1:
#         print(a[0, 0])
#     else:
#         q = cut_permutation_matrix(a)
#         return(leave_behind(np.matmul(q, a)))

# leave_behind(a)

# # nothing to do with things above
# from math import pi
# import numpy as np
# count = 0
# times = 10 ** 6
# for _ in range(times):
#     x, y=np.random.random((2))
#     square_distance = x ** 2  + y ** 2
#     if square_distance <= 1:
#         count += 1
# similar_pi = 4 * count/times
# print(similar_pi)