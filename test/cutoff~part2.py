import numpy as np
import copy

# forbidden solution
patterns = [[1, 0, 17], [1, 6, 14], [1, 12, 11], [1, 18, 8], [3, 4, 9], [3, 10, 6], [12, 2, 3]]
L = {0: 4100, 1: 4350, 2: 4700}
propatterns = [[0 for _ in range(len(patterns[0]))]] + patterns # the first space is used for doing nothing
print(propatterns)
need = np.array([114, 246, 174])
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

def solve1(depth):
    
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

              

result = solve1(25)

print(result)
ac = calcu(result)
print(ac)
print(need - ac)
