import numpy as np
import random

faces = {"fornt":0, "back":1, "left":2, "right":3, "top":4, "bottom":5}

# 平面旋转矩阵
def rotate(faceMatrix, axis="z", clockwise=True):
    if axis=="z":
        if clockwise:
            faceMatrix[0,0],faceMatrix[0,1],faceMatrix[1,0],faceMatrix[1,1] = \
                faceMatrix[1,0],faceMatrix[0,0],faceMatrix[1,1],faceMatrix[0,1]
        else:
            faceMatrix[0,0],faceMatrix[0,1],faceMatrix[1,0],faceMatrix[1,1] = \
                faceMatrix[0,1],faceMatrix[1,1],faceMatrix[0,0],faceMatrix[1,0]
    elif axis=="y":
        faceMatrix[0,0],faceMatrix[0,1],faceMatrix[1,0],faceMatrix[1,1] = \
            faceMatrix[0,1],faceMatrix[0,0],faceMatrix[1,1],faceMatrix[1,0]
    elif axis=="x":
        faceMatrix[0,0],faceMatrix[0,1],faceMatrix[1,0],faceMatrix[1,1] = \
            faceMatrix[1,0],faceMatrix[1,1],faceMatrix[0,0],faceMatrix[0,1]


# 旋转整个魔方，不绕z轴旋转，只x，y轴旋转
#             00 01  
#             10 11
#       00 01 00 01 00 01   00 01
#       10 11 10 11 10 11   10 11
#             00 01
#             10 11
# axis = ["x","y"]
def rotatecube(cube, axis, clockwise=True):
    if clockwise:
        if axis=="x":
            cube[[faces["fornt"],faces["bottom"],faces["back"],faces["top"]]] = \
                cube[[faces["top"],faces["fornt"],faces["bottom"],faces["back"]]]            
            rotate(cube[faces["top"]], axis="x")
            rotate(cube[faces["back"]], axis="x")
            rotate(cube[faces["left"]])
            rotate(cube[faces["right"]], clockwise=False)
        elif axis=="y":
            cube[[faces["fornt"],faces["right"],faces["back"],faces["left"]]] = \
                cube[[faces["right"],faces["back"],faces["left"],faces["fornt"]]]
            rotate(cube[faces["right"]], axis="y")
            rotate(cube[faces["back"]], axis="y")
            rotate(cube[faces["top"]], clockwise=True)
            rotate(cube[faces["bottom"]], clockwise=False)
    else:
        if axis=="x":
            cube[[faces["fornt"],faces["bottom"],faces["back"],faces["top"]]] = \
                cube[[faces["bottom"],faces["back"],faces["top"],faces["fornt"]]]
            rotate(cube[faces["bottom"]], axis="x")
            rotate(cube[faces["back"]], axis="x")            
            rotate(cube[faces["left"]], clockwise=False)
            rotate(cube[faces["right"]])
        elif axis=="y":
            cube[[faces["fornt"],faces["right"],faces["back"],faces["left"]]] = \
                cube[[faces["left"],faces["fornt"],faces["right"],faces["back"]]]
            rotate(cube[faces["left"]], axis="y")
            rotate(cube[faces["back"]], axis="y")
            rotate(cube[faces["top"]], clockwise=False)
            rotate(cube[faces["bottom"]], clockwise=True)


# 旋转魔方的面
def rotateface(cube, facename, clockwise=True):
    if facename=="bottom":
        rotatecube(cube, axis="x",clockwise=False)
    elif facename=="top":
        rotatecube(cube, axis="x",clockwise=True)
    elif facename=="back":
        rotatecube(cube, axis="x",clockwise=True)
        rotatecube(cube, axis="x",clockwise=True)
    elif facename=="left":
        rotatecube(cube, axis="y",clockwise=False)
    elif facename=="right":
        rotatecube(cube, axis="y",clockwise=True)

    face = cube[faces["fornt"]]
    rotate(face, axis="z", clockwise=clockwise)

    if clockwise:
        cube[faces["top"]][1,0],cube[faces["top"]][1,1],cube[faces["right"]][0,0],cube[faces["right"]][1,0], \
        cube[faces["bottom"]][0,0],cube[faces["bottom"]][0,1],cube[faces["left"]][0,1],cube[faces["left"]][1,1] =  \
            cube[faces["left"]][1,1],cube[faces["left"]][0,1],cube[faces["top"]][1,0],cube[faces["top"]][1,1], \
            cube[faces["right"]][1,0],cube[faces["right"]][0,0],cube[faces["bottom"]][0,0],cube[faces["bottom"]][0,1] 
    else:
        cube[faces["top"]][1,0],cube[faces["top"]][1,1],cube[faces["right"]][0,0],cube[faces["right"]][1,0], \
        cube[faces["bottom"]][0,0],cube[faces["bottom"]][0,1],cube[faces["left"]][0,1],cube[faces["left"]][1,1] =  \
            cube[faces["right"]][0,0],cube[faces["right"]][1,0],cube[faces["bottom"]][0,1],cube[faces["bottom"]][0,0], \
            cube[faces["left"]][0,1],cube[faces["left"]][1,1],cube[faces["top"]][1,1],cube[faces["top"]][1,0] 

    if facename=="bottom":
        rotatecube(cube,axis="x",clockwise=True)
    elif facename=="top":
        rotatecube(cube,axis="x",clockwise=False)
    elif facename=="back":
        rotatecube(cube,axis="x",clockwise=False)
        rotatecube(cube,axis="x",clockwise=False)
    elif facename=="left":
        rotatecube(cube,axis="y",clockwise=True)
    elif facename=="right":
        rotatecube(cube,axis="y",clockwise=False)

def printcube(cube):
    print("   ",cube[faces["top"]][0,0],cube[faces["top"]][0,1])
    print("   ",cube[faces["top"]][1,0],cube[faces["top"]][1,1])
    print(cube[faces["left"]][0,0],cube[faces["left"]][0,1],cube[faces["fornt"]][0,0],cube[faces["fornt"]][0,1],\
          cube[faces["right"]][0,0],cube[faces["right"]][0,1],cube[faces["back"]][0,0],cube[faces["back"]][0,1])    
    print(cube[faces["left"]][1,0],cube[faces["left"]][1,1],cube[faces["fornt"]][1,0],cube[faces["fornt"]][1,1],\
          cube[faces["right"]][1,0],cube[faces["right"]][1,1],cube[faces["back"]][1,0],cube[faces["back"]][1,1])
    print("   ",cube[faces["bottom"]][0,0],cube[faces["bottom"]][0,1])
    print("   ",cube[faces["bottom"]][1,0],cube[faces["bottom"]][1,1])
    print()

# 创建一个新魔方 
cube = np.zeros((6,2,2),dtype=int)
for i in range(6):
    cube[i]+=i

# 随机100步打乱
for i in range(100):
    facename = random.choice(list(faces.keys()))
    clockwise = random.random()>0.5
    rotateface(cube, facename, clockwise)

printcube(cube)
print("Start ....")

def checkbottom_same(cube):
    for fn in ["fornt","left","right","back"]:
        if not cube[faces[fn]][1,0]==cube[faces[fn]][1,1]:
            return False
    return True

def checkoneface_same(cube):
    if not checkbottom_same(cube): 
        return False

    for fn in ["fornt","left","right","back"]:
        if (cube[faces[fn]] == cube[faces[fn]][1,0]).all():
            return True
        
    return False

def check_same(cube):
    if not checkoneface_same(cube):
        return False
    
    for fn in ["fornt","left","right","back"]:
        if not (cube[faces[fn]]==cube[faces[fn]][1,0]).all():
            return False
    return True

for i in range(10000):
    facename = random.choice(list(faces.keys()))
    clockwise = random.random()>0.5
    rotateface(cube, facename, clockwise)
    if checkbottom_same(cube):
        print("OKey, Find One !!!")
        break
printcube(cube)

for i in range(10000):
    _cube = np.copy(cube)
    find = False
    for i in range(20):
        facename = random.choice(list(faces.keys()))
        clockwise = random.random()>0.5
        rotateface(_cube, facename, clockwise)
        if checkoneface_same(_cube):
            print("OKey, Find Two !!!")
            find = True
            break
    if find: break
printcube(_cube)

for i in range(10000):
    __cube = np.copy(_cube)
    find = False
    for i in range(20):
        facename = random.choice(list(faces.keys()))
        clockwise = random.random()>0.5
        rotateface(__cube, facename, clockwise)
        if check_same(__cube):
            print("OKey, Find !!!")
            find = True
            break
    if find: break

printcube(__cube)    