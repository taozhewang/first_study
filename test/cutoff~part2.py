import numpy as np
import copy

# forbidden solution
patterns_set = [([1, 0, 17], [0, 7, 17, 6]), ([1, 6, 14], [0, 8, 20, 7]), ([1, 12, 11], [0, 9, 23, 8]),
                ([1, 18, 8], [0, 10, 26, 9]), ([3, 4, 9], [0, 6, 15, 5]), ([3, 10, 6], [0, 7, 18, 6]),
                ([3, 16, 3], [0, 8, 21, 7]), ([3, 22, 0], [0, 9, 24, 8]), ([5, 2, 4], [0, 4, 10, 3]), 
                ([5, 8, 1], [0, 5, 13, 4]), ([8, 0, 16], [0, 9, 23, 8]), ([12, 2, 3], [0, 6, 16, 5]),
                ([12, 8, 0], [0, 7, 19, 6]), ([19, 2, 2], [0, 8, 22, 7]), ([26, 2, 1], [0, 10, 28, 9])]
patterns = [p[0] for p in patterns_set]
patterns_property = [p[1] for p in patterns_set]
# patterns = [[1, 0, 17], [1, 6, 14], [1, 12, 11], [1, 18, 8], [3, 4, 9], [3, 10, 6], [12, 2, 3]]
L = {0: 4100, 1: 4350, 2: 4700}
l = 12000


propatterns = [[0 for _ in range(len(patterns[0]))]] + patterns # the first space is used for doing nothing
print(propatterns)
proproperty = [[0 for _ in range(len(patterns_property[0]))]] + patterns_property
need = np.array([852, 658, 162])
'''
accumulate: [1, 2, 3, ... , 2]
             number of used patterns
patterns: [[5, 2, 4],
           [4, 8, 9], 
           [7, 3, 2],
           ...
           [10, 2, 0]]


             

             '''
def calcu(accumulate):
    lvalues = L.values()
    count = np.zeros(len(lvalues))
    for i, num in enumerate(accumulate):
        count += np.array(propatterns[i]) * num
    return count

def calc1(depth):
    
    accumulate = np.zeros(len(patterns) + 1) # the first space is used for doing nothing
    backward_forbid = []
    forward_forbid = []
    forbid_length = 10
    maxleft = np.sum(need ** 2) + 1
    minleft = maxleft
    
    # decrease then increase
    while True:
        print('////////////////////////////////')
        cross = np.ones((len(propatterns), len(propatterns))) * maxleft # collect all potential left
        for i in range(len(propatterns)):
            accumulate[0] = 50
            if accumulate[i] > 0:
                Re_accumulate = copy.deepcopy(accumulate)
                Re_accumulate[i] -= 1
                for j in range(len(propatterns)):
                    Re_Re_accumulate = copy.deepcopy(Re_accumulate)
                    Re_Re_accumulate[j] += 1
                    left_list = need - calcu(Re_Re_accumulate)
                    left = np.sum(left_list ** 2)

                    # print('current issue:', (i, j), left)
                    if all(left_list >= 0):
                        if left < minleft:
                            cross[i, j] = left
                        # elif i not in forward_forbid or j not in backward_forbid:
                        elif (i, j) not in list(zip(forward_forbid, backward_forbid)):
                            cross[i, j] = left
                        else:
                            cross[i, j] = maxleft
                    else:
                        cross[i, j] = maxleft
        for i in range(len(propatterns)):
            cross[i, i] = maxleft
        # print(cross)

        currmin = np.min(cross)
        loc_list = list(np.where(cross == currmin))
        loc = list(zip(loc_list[0], loc_list[1]))
        x, y = loc[0]
        accumulate[x] -= 1
        accumulate[y] += 1
        
        forward_forbid.append(x)
        backward_forbid.append(y)
        if len(forward_forbid) > forbid_length:
            forward_forbid.pop(0)
        if len(backward_forbid) > forbid_length:
            backward_forbid.pop(0)

        if currmin < minleft:
            minleft = currmin
            minaccum = copy.deepcopy(accumulate)

        depth -= 1
        if depth == 0:
            return minaccum
        print('depth:', depth)
        print('curr_stage:', accumulate, currmin)

              

result = calc1(65)

print(result)
ac = calcu(result)
print(ac)
print(need - ac)

# # from the solution to the local best solution

# # def solve2(depth):
# '''[([1, 0, 17], [0, 7, 17, 6]), ([1, 6, 14], [0, 8, 20, 7]),
# ([1, 12, 11], [0, 9, 23, 8]), ([1, 18, 8], [0, 10, 26, 9]),
# ([3, 4, 9], [0, 6, 15, 5]), ([3, 10, 6], [0, 7, 18, 6]),
# ([3, 16, 3], [0, 8, 21, 7]), ([3, 22, 0], [0, 9, 24, 8]),
# ([5, 2, 4], [0, 4, 10, 3]), ([5, 8, 1], [0, 5, 13, 4]),
# ([8, 0, 16], [0, 9, 23, 8]), ([12, 2, 3], [0, 6, 16, 5]),
# ([12, 8, 0], [0, 7, 19, 6]), ([19, 2, 2], [0, 8, 22, 7]),
# ([26, 2, 1], [0, 10, 28, 9])]


# '''

# def calc2(result, depth):
#     def crudecompute(left, l, L):

        
#         # print(left)

#         # available_left = []
#         # close = []
#         # pro = []
#         # for i, pattern in enumerate(patterns):
#         #     weak_left = left - pattern
#         #     if all(weak_left >= 0):
#         #         available_left.append(weak_left)
#         #         close.append(np.sum(weak_left))
#         #         pro.append(i)
#         # if len(available_left)> 0:
#         #     m = np.min(close)
#         #     w = np.where(close == m)
#         #     loc = w[0][0]
#         #     left = available_left[loc]
#         #     minus = patterns_property[pro[loc]]
#         cache = []    
#         while True:
#             a = 0
#             for i, pattern in enumerate(patterns):
#                 curr_pattern = left - pattern
#                 if all(curr_pattern >= 0):
#                     left = curr_pattern
#                     cache.append(patterns_property[i])
#                     a = 1
#                     break
#             if a == 0:
#                 break


#         def cuttree(tree):
#             newtree = []
#             for branch in tree:
#                 if branch not in newtree:
#                     newtree.append(branch)
#             return newtree
#         def doublecut(tree, journal):
#             newtree = []
#             newjournal = []
#             for i, branch in enumerate(tree):
#                 if branch not in newtree:
#                     newtree.append(branch)
#                     newjournal.append(journal[i])
#             return newtree, newjournal
#         height = 0
#         for i, num in enumerate(left):
#             height += num * L[i]
#         gauge = height // l + bool(height % l)
#         tree = [[]]
#         journal = [left]
#         # print(left)

        

#         while True:

#             if np.sum(left) > 20:
#                 for i, num in enumerate(left):
#                     tree[0] += [i] * int(num)
#                 break
#             print('////////////////////////////////////////////////////////////////')
#             newtree = []
#             newjournal = []
#             for i, branch in enumerate(tree):
#                 newleft = journal[i]
#                 for j in range(len(newleft)):
#                     if newleft[j] == 0:
#                         # newtree.append(branch)
#                         # newjournal.append(newleft)
#                         continue
#                     else:
#                         newbranch = copy.deepcopy(branch)
#                         newbranch.append(j)
#                         newtree.append(newbranch)
#                         nl = copy.deepcopy(newleft)
#                         nl[j] -= 1
#                         newjournal.append(nl)
#             tree, journal = doublecut(newtree, newjournal)
            
#             b = 1
#             for newleft in journal:
#                 if any(newleft):
#                     b = 0
#             if b:
#                 break    

#         # def buildtree(left, branch, pointer):
#         #     leftR = copy.deepcopy(left)
#         #     branchR = copy.deepcopy(branch)
            
#         #     if all(leftR == 0):
#         #         tree.append(branch)
#         #         return
#         #     elif left[pointer] == 0:
#         #         return
#         #     else:
#         #         branchR.append(pointer)
#         #         expandtree(leftR, branchR)

#         # def expandtree(left, branch):
#         #     for pointer in range(len(left)):
#         #         buildtree(left, branch, pointer)
       
#         # expandtree(left, [])
#         tree = cuttree(tree)
#         journal1 = []
#         journal2 = []
#         # print(tree)
#         for branch in tree:
            
#             height = 0
#             count = 0
#             space = gauge
#             for leaf in branch:
#                 # print(leaf)
#                 height += L[leaf]
#                 count += 1
#                 if height > l:
#                     height = L[leaf]
#                     space -= 1
#                     # print(space)
#                     if space == 0:
#                         count -= 1
#                         paste = len(branch) - count
#                         cut = len(branch) - 1 + paste
#                         # print(paste)
#                         break
#                     elif count == len(branch):
#                         paste = 0
#                         cut = len(branch) - 1
#                         break
#                 elif count == len(branch):
#                     paste = 0
#                     cut = len(branch) - 1
#                     break
#             journal1.append(paste)
#             journal2.append(cut)
#         paste = np.min(journal1)
#         cut = np.min(journal2)

#         if len(cache) > 0:
#             for i in cache:
#                 cut += i[2]
#                 paste += i[3]
#         return paste, cut

#     accumulate = result # the first space is used for doing nothing
#     backward_forbid = []
#     forward_forbid = []
#     forbid_length = 10
#     maxcut = np.sum(need) * 2
#     maxpaste = np.sum(need) * 2
#     mincut = maxcut
#     minpaste = maxpaste
#     cutweight = 1
#     pasteweight = 1
#     maxmeasure = mincut * cutweight + minpaste * pasteweight
#     minmeasure = maxmeasure

    
#     # decrease then increase
#     while True:
#         print('////////////////////////////////')
#         cross = np.ones((len(propatterns), len(propatterns))) * maxmeasure
#      # collect all potential left
#         for i in range(len(propatterns)):
#             accumulate[0] = 50
#             if accumulate[i] > 0:
#                 Re_accumulate = copy.deepcopy(accumulate)
#                 Re_accumulate[i] -= 1
#                 for j in range(len(propatterns)):
#                     Re_Re_accumulate = copy.deepcopy(Re_accumulate)
#                     Re_Re_accumulate[j] += 1
                    
#                     cuttime = 0
#                     pastetime = 0
#                     for i, num in enumerate(Re_Re_accumulate):
#                         cuttime += proproperty[i][2] * num
#                         pastetime += proproperty[i][3] * num
                    
#                     left = need - calcu(Re_Re_accumulate)
#                     if all(left >= 0):
#                         leftcut, leftpaste = crudecompute(left, l, L)
#                         cuttime += leftcut
#                         pastetime += leftpaste
#                         cuttime = cuttime * cutweight
#                         pastetime = pastetime * pasteweight
#                         measure = cuttime + pastetime
                    

#                     # print('current issue:', (i, j), left)
                    
#                         if measure < minmeasure:
#                             cross[i, j] = measure
#                         # elif i not in forward_forbid or j not in backward_forbid:
#                         elif (i, j) not in list(zip(forward_forbid, backward_forbid)):
#                             cross[i, j] = measure
#                         else:
#                             cross[i, j] = maxmeasure
#                     else:
#                         cross[i, j] = maxmeasure
#         for i in range(len(propatterns)):
#             cross[i, i] = maxmeasure
#         # print(cross)

#         currmin = np.min(cross)
#         loc_list = list(np.where(cross == currmin))
#         loc = list(zip(loc_list[0], loc_list[1]))
#         x, y = loc[0]
#         accumulate[x] -= 1
#         accumulate[y] += 1
        
#         forward_forbid.append(x)
#         backward_forbid.append(y)
#         if len(forward_forbid) > forbid_length:
#             forward_forbid.pop(0)
#         if len(backward_forbid) > forbid_length:
#             backward_forbid.pop(0)

#         if currmin < minmeasure:
#             minmeasure = currmin
#             minaccum = copy.deepcopy(accumulate)

#         depth -= 1
#         if depth == 0:
#             return minaccum
#         print('depth:', depth)
#         print('curr_stage:', accumulate, currmin)

# result2 = calc2(result, 50)
# print(result2)