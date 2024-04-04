# import numpy as np

# coins = [1, 2, 8]
# target = 19

# def m(coins, target):
#     def countable(coins, num, pointer):
#         if pointer == len(coins):
#             return np.array([num])
#         elif num < coins[pointer]:
#             return np.array([num])
#         else:
#             return np.concatenate((countable(coins, num - coins[pointer], pointer + 1),
#                                 countable(coins, num, pointer + 1)), axis = None)
#     def seek_countable(coins):
#         return countable(coins, sum(coins), 0)
#     a = seek_countable(coins)
#     b = np.arange(1, target + 1)
#     c = np.setdiff1d(b, a)
#     if len(c) == 0:
#         assert 1 == 0, f'we find it!{coins}'

# # dim 1
# for i in range(1, target + 1):
#     p = coins + [i]
#     m(p, target)
# # dim 2
# for i in range(1, target + 1):
#     for j in range(1, target + 1):
#         p = coins + [i, j]
#         m(p, target)

# not yet