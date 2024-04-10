import numpy as np
import random
import hashlib
import json,os,gzip

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

# 检查匹配的程度
def checksame(cube):
    result= 0
    for facename in faces:
        id = faces[facename]
        if cube[id][0,0]==id: result+=1
        if cube[id][0,1]==id: result+=1
        if cube[id][1,0]==id: result+=1
        if cube[id][1,1]==id: result+=1
    return result

# 魔方的hash值
def hash(cube):
    flattened_matrix = cube.flatten()
    hash_object = hashlib.sha256()
    hash_object.update(flattened_matrix)
    hash_value = hash_object.hexdigest()    
    return hash_value


# 创建一个新魔方 
cube = np.zeros((6,2,2),dtype=int)
for facename in faces:
    cube[faces[facename]]+=faces[facename]
# printcube(cube)

# 创建步骤cache，里面直接存放还原步骤:
# 格式 {"hash": [act9 .... act0] act 为（facename, clockwise）
cache_file="game/cube/cube2.json.gz"
if os.path.exists(cache_file):
    print("loading cache")
    with gzip.open(cache_file, "rt") as f:
        cache=json.load(f)
else:
    print("building cache")
    cache={}
    for i in range(2000000):
        if i%20000==0: 
            print("%d%%"%(i*100/2000000))
        _cube = np.copy(cube)
        act = []
        for _ in range(10):
            facename = random.choice(list(faces.keys()))
            clockwise = random.random()>0.5
            rotateface(_cube, facename, clockwise)
            act.insert(0,(facename, not clockwise))
        h = hash(_cube)
        if h not in cache: 
            cache[h]=act
        else:
            if len(cache[h])>len(act):
                cache[h]=act
    with gzip.open(cache_file, "wt") as f:
        json.dump(cache, f)        
print("load cache end, size:",len(cache))

# 随机100步打乱
for i in range(100):
    facename = random.choice(list(faces.keys()))
    clockwise = random.random()>0.5
    rotateface(cube, facename, clockwise)

printcube(cube)
print("Start ....")

find=False
max_count = 0
act = []
for _ in range(100000):
    _cube = np.copy(cube)
    _act = [a for a in act]
    for _ in range(20):
        facename = random.choice(list(faces.keys()))
        clockwise = random.random()>0.5
        rotateface(_cube, facename, clockwise)
        _act.append((facename, clockwise))
        h = hash(_cube)
        if h in cache:
            print("cache find !!!")
            find=True
            printcube(_cube)
            for facename, clockwise in cache[h]:
                rotateface(_cube, facename, clockwise)
                _act.append((facename, clockwise))
            cube = np.copy(_cube)
            act = _act
            break
        _max_count = checksame(_cube)
        if _max_count>max_count:
            cube = np.copy(_cube)
            max_count = _max_count
            print(max_count)
            printcube(cube)
            act = _act
            if max_count==24: 
                print("find !!!")
                find=True
            break
    if find: break

if not find:
    print("fail .")
else:
    print(len(act),act)
    printcube(cube)


