import copy
import numpy as np
fill = input('fill in data or by default: press 0 or 1: ')
if fill == '0':
    l = int(input('raw material length: '))
    n = int(input('the number of objects: '))
    L = {}
    for i in range(n):
        L[i] = int(input(f'object L{i + 1}: '))
    radius = int(input('radius of the number of raw materials: '))
    losses1 = int(input('max left of patterns: '))
else:
    l = 12000
    L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
    radius = 10
    losses1 = 0
joint = 200
L = list(L.values())
'''
left = [1, 3, 6]
'''
# def calc_totallength(left, L):
#     totallength = 0
#     for i in range(len(left)):
#         totallength += left[i] * L[i]
#     return totallength

# def terms_origin(N):
#     # N = [3, 4, 1, 2]
#     P = []
#     for i in N:
#         P.append(int(i))
#     N = P
#     n = len(N)
#     terms = []
#     lefft = []
    
#     j = 0
#     for i in range(n):
#         if N[i] > 0:
#             K = copy.deepcopy(N)
#             terms.append([i])
#             lefft.append(K)
#             lefft[j][i] -= 1
#             j += 1
    
#     print(terms)
#     print(lefft)
#     while True:
#         # print('/////////////////////////////////////')
#         # print(terms)
#         # print(lefft)
#         if not any(lefft[0]):
#             break 
#         new_terms = copy.deepcopy(terms)
#         new_lefft = copy.deepcopy(lefft)
#         for i in range(len(terms)):
#             t = new_terms.pop(0)
#             l = new_lefft.pop(0)
#             for j in range(n):
#                 if l[j] > 0:
#                     T = copy.deepcopy(t)
#                     T.append(j)
#                     L = copy.deepcopy(l)
#                     L[j] -= 1
#                     new_terms.append(T)
#                     new_lefft.append(L)
#         terms = new_terms
#         lefft = new_lefft
#     return terms

# # t = terms_origin([0., 7., 2., 1.])
# # print(t)
# # print(len(t))

# def calc_left(left):
#     totallength = calc_totallength(left, L)
#     number_l = int(totallength / l) + 1
#     possibility = terms_origin(left)
#     for path in possibility:
#         accumulate_length = 0
#         i = 0
#         u = number_l
#         use = [[]]
#         stage = 0
#         joint_left = []
#         while True:
#             if i == len(path):
#                 break
#             if u == 0:
#                 break
#             accumulate_length += L[path[i]]
#             if accumulate_length > l:
#                 joint_left.append(L[path[i]] - (accumulate_length - l))
#                 u -= 1
#                 accumulate_length = 0
#                 stage += 1
#             elif accumulate_length < l and accumulate_length > l - joint:
#                 # warning: there does NOT exist an object whose length is less than the joint!
#                 u -= 1
#                 accumulate_length = 0
#                 use[stage].append(path[i])
#                 i += 1
#                 stage += 1
#                 use[stage] = 1
#             else:
#                 use[stage].append(path[i])
#                 i += 1
#         p2 = path[i:]

terms1 = []
terms2 = []
terms3 = []
terms4 = []
use = []
paste = []

def calc_left(left, use, paste, accumulate_length, length, stage, joint):
    
    if not any(left):
        clength = copy.deepcopy(length)
        clength[stage].append(l - accumulate_length)

        terms1.append(use)
        terms2.append(paste)
        terms4.append(clength)
        terms1.append('//')
        terms2.append('//')
        terms4.append('//')
        return 
    for i in range(len(left)):
        if left[i] > 0:
            cleft = copy.deepcopy(left)
            cuse = copy.deepcopy(use)
            cpaste = copy.deepcopy(paste)
            caccumulate_length = accumulate_length + L[i]
            cstage = stage
            clength = copy.deepcopy(length)
            if caccumulate_length > l + joint:
                caccumulate_length -= l
                cleft[i] -= 1
                cpaste.append(i)
                clength[cstage].append(L[i] - caccumulate_length)
                cstage += 1
                clength.append([caccumulate_length])
                cuse.append([])
                calc_left(cleft, cuse, cpaste, caccumulate_length, clength, cstage, joint)
            elif caccumulate_length > l and caccumulate_length <= l + joint:
                caccumulate_length -= l
                cleft[i] -= 1
                cpaste.append(i)
                clength[cstage].append(joint - caccumulate_length)
                clength[cstage].append(L[i] - joint)
                cstage += 1
                clength.append([joint])
                cuse.append([])
                calc_left(cleft, cuse, cpaste, joint, clength, cstage, joint)
            elif caccumulate_length < l and caccumulate_length > l - joint:
                cleft[i] -= 1
                cuse[cstage].append(i)
                clength[cstage].append(L[i])
                clength[cstage].append( - caccumulate_length + l)
                cstage += 1
                clength.append([])
                cuse.append([])
                cpaste.append(-1)
                calc_left(cleft, cuse, cpaste, 0, clength, cstage, joint)
            else:
                cleft[i] -= 1
                print(L[i])
                clength[cstage].append(L[i])
                print(clength)
                cuse[cstage].append(i)
                calc_left(cleft, cuse, cpaste, caccumulate_length, clength, cstage, joint)

calc_left([3, 2, 2], [[]], [], 0, [[]], 0, joint)

print(terms1)
print(terms2)
print(terms4)