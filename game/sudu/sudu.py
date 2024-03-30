import random

def generate_sudoku():
    # 创建一个9x9的二维列表，并初始化为0
    sudoku = [[0] * 9 for _ in range(9)]
    
    # 调用递归函数填充数独
    fill_sudoku(sudoku, 0, 0)
    
    return sudoku

def fill_sudoku(sudoku, row, col):
    if col >= 9:  # 列越界，换到下一行
        col = 0
        row += 1
        if row >= 9:  # 行越界，数独已填满
            return True
    
    # 在当前位置尝试填入数字
    for num in random.sample(range(1, 10), 9):  # 随机打乱1到9的数字顺序
        if is_valid(sudoku, row, col, num):
            sudoku[row][col] = num
            
            if fill_sudoku(sudoku, row, col + 1):  # 递归调用填充下一列
                return True
            
            sudoku[row][col] = 0  # 回溯，重置当前位置为0
    
    return False

def is_valid(sudoku, row, col, num):
    # 检查当前数字在所在行是否重复
    for i in range(9):
        if sudoku[row][i] == num:
            return False
    
    # 检查当前数字在所在列是否重复
    for i in range(9):
        if sudoku[i][col] == num:
            return False
    
    # 检查当前数字在所在3x3小方格是否重复
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if sudoku[start_row + i][start_col + j] == num:
                return False
    
    return True

# 生成数独
sudoku = generate_sudoku()

# 打印数独
for row in sudoku:
    print(row)