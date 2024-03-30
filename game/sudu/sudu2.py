import random

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

def generate_sudoku(difficulty):
    sudoku = [[0] * 9 for _ in range(9)]
    fill_sudoku(sudoku, 0, 0)
    remove_numbers(sudoku, difficulty)
    return sudoku

def fill_sudoku(sudoku, row, col):
    if col >= 9:
        col = 0
        row += 1
        if row >= 9:
            return True

    nums = random.sample(range(1, 10), 9)
    for num in nums:
        if is_valid(sudoku, row, col, num):
            sudoku[row][col] = num
            if fill_sudoku(sudoku, row, col + 1):
                return True
            sudoku[row][col] = 0

    return False

def remove_numbers(sudoku, difficulty):
    # 根据难度级别决定要移除的数字数量
    if difficulty == 'easy':
        num_to_remove = random.randint(30, 40)
    elif difficulty == 'medium':
        num_to_remove = random.randint(40, 50)
    elif difficulty == 'hard':
        num_to_remove = random.randint(50, 60)
    else:
        num_to_remove = random.randint(60, 70)

    # 随机移除数字
    while num_to_remove > 0:
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        if sudoku[row][col] != 0:
            sudoku[row][col] = 0
            num_to_remove -= 1

# 生成数独谜题
difficulty_level = 'medium'  # 设置难度级别，可以选择 'easy', 'medium', 'hard' 或自定义
sudoku = generate_sudoku(difficulty_level)

# 打印数独谜题
for row in sudoku:
    print(row)