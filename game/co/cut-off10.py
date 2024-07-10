import numpy as np
import random
import time

'''
用蚁群求解钢筋切割问题
参考： https://github.com/guofei9987/scikit-opt/blob/master/sko/ACA.py
'''

# 原始钢筋长度
l = 12000
# 最小可接长度,小于200的部分会做为废料
l_min = 200
# 钢筋的规格
l_size = 32
# 目标钢筋长度
L = {'L1' : 4100, 'L2' : 4350, 'L3' : 4700}
# 目标钢筋的数量
need = np.array([552, 658, 462],dtype=int)

# ABC参数
colony_size = 100
max_iterations = 100000
limit = 100

# 计算解的适应度值(成本)
def calculate_fitness(solution):
    loss, joint, cost, group_count, _, _ = evaluate(solution[::-1], l, l_min, l_size)
    # return cost
    # return 1/group_count
    return cost+1/group_count

# 计算组合的损失、接头、成本、小组个数、最后一组的位置
def evaluate(solution, l, l_min, l_size):
    _l = l
    group_count = 0
    group_endpos = 0
    loss = 0
    joint = 0
    cost = 0

    for i,num in enumerate(solution):
        while _l< num:
            _l += l
            joint += 1
        
        _l -= num

        if _l < l_min:
            loss += _l
            group_count += 1
            group_endpos = i+1
            _l = l 

    loss += _l
    cost_param = 0.00617 * 2000 * (l_size ** 2) / 1000
    cost = loss * cost_param + joint * 10
    return loss, joint, cost, group_count, 0, group_endpos


class ACA_TSP:
    def __init__(self, func, n_dim,
                 reference_matrix,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 alpha=1, beta=2, rho=0.1,
                 ):
        self.func = func
        self.n_dim = n_dim  # 城市数量
        self.size_pop = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度
        self.reference_matrix = reference_matrix    # 参考矩阵

        if distance_matrix is None: # 先验概率矩阵
            self.prob_matrix_distance = 1 
        else:
            self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))  # 避免除零错误

        self.Tau = np.ones((n_dim, n_dim))  # 信息素矩阵，每次迭代都会更新

        self.Table = np.zeros((size_pop, n_dim), dtype=int)  # 某一代每个蚂蚁的爬行路径

        self.y = None  # 某一代每个蚂蚁的爬行总距离
        self.generation_best_X, self.generation_best_Y = [], []  # 记录各代的最佳情况
        self.best_x, self.best_y = None, None

    def getFirstIdx(self, allow_list):
        has_v = []
        result = []
        for i in allow_list:
            if self.reference_matrix[i] not in has_v:
                result.append(i)
                has_v.append(self.reference_matrix[i])
        return result
    
    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
                  
        for i in range(self.max_iter):  # 对每次迭代
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta  # 转移概率，无须归一化。
            for j in range(self.size_pop):  # 对每个蚂蚁
                allow_list = list(range(self.n_dim))
                for k in range(self.n_dim):  # 蚂蚁到达的每个节点
                    _allow_list= self.getFirstIdx(allow_list)
                    
                    prob = prob_matrix[self.Table[j, k], _allow_list]                    
                    prob = prob / prob.sum()  # 概率归一化
                    next_point = np.random.choice(_allow_list, p=prob)
                    # dirichlet = np.random.dirichlet(3 * np.ones(len(_allow_list)))
                    # p=0.75                    
                    # next_point = np.random.choice(_allow_list, p=prob*p+dirichlet*(1-p))
                    self.Table[j, k] = next_point
                    allow_list.remove(next_point)

            # 计算距离
            y = np.array([self.func(np.take(self.reference_matrix, i)) for i in self.Table])

            # 顺便记录历史最好情况
            avg_y = np.mean(y)
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # 计算需要新涂抹的信息素
            # delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # 每个蚂蚁
                v = 1/y[j] # 蚂蚁的适应度
                for k in range(self.n_dim):  # 每个节点
                    n = self.Table[j, k] # 蚂蚁到达n1节点
                    self.Tau[k, n] = (1-self.rho) * self.Tau[k,n] + v*(1-self.rho)**k
                    # self.Tau[k,n] = (1-self.rho) * self.Tau[k,n] + 1/y[j] # 信息素飘散
                    # delta_tau[k, n] += 1 / y[j] 
                    # v = self.reference_matrix[n] # 节点v的值
                    # for m in range(self.n_dim):  # 每个节点
                    #     if v == self.reference_matrix[m]:
                    #         delta_tau[k, m] += 1 / y[j]  # 涂抹的信息素

            # 信息素飘散+信息素涂抹
            # self.Tau = (1 - self.rho) * self.Tau + delta_tau

            print(i, np.min(self.generation_best_Y) , avg_y, )
            if i%10==0:   
                # out=np.take(self.reference_matrix, x_best)              
                # print(*x_best)
                print(x_best)

        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y

    fit = run

best_solution, best_fitness = None, np.inf

points_coordinate = []
for i,key in enumerate(L):
    points_coordinate += [L[key]] * need[i]  
points_coordinate = np.array(points_coordinate)


start_time = time.time()
aca_tsp=ACA_TSP(func=calculate_fitness, n_dim=np.sum(need), reference_matrix=points_coordinate, rho=1e-3, size_pop=100)
best_solution, best_fitness = aca_tsp.run(max_iterations)

out = [0, 0, 0]
L_values = [L[key] for key in L]
for num in best_solution:
    out[L_values.index(num)] += 1
print("验算结果:", out, "实际需求:", need)

best_loss, best_joints, best_cost, _, _, _ = evaluate(best_solution, l, l_min, l_size)
print("最佳方案为:")
print(best_solution)
print("废料长度:", best_loss)
print("接头数量:", best_joints)
print("总成本:", best_cost)
print(f"用时: {time.time() - start_time} 秒")