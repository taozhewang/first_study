# Datatypes, Control Flows, Functions
def count_five_or_seven():
    for i in range(1000):
        if (i + 1) % 5 ==0 or (i + 1) % 7 == 0:
            print(i)

def print_S0():
    def row():
        for _ in range(3):
            print('o' * 17)
    def column(a):
        row()
        if a:
            for _ in range(3):
                print(' ' * 13, 'o'* 4, sep = '')
            row()
        else:
            for _ in range(3):
                print('o'* 4)
    for i in range(2):
        column(i)

def search_vegetable(vegetfruit):
    vegetable = {}
    for i in range(len(vegetfruit)):
        if vegetfruit[i] == "cabage" or "potato":
            vegetable[i] = vegetfruit[i]
    return vegetable
def inverse_list(your_list):
    a = []
    for i in range(len(your_list) - 1, -1, -1):
        a.append(your_list[i])
    return a
def days_month(month):
    month = month.strip().lower()
    year = {'january': 31, 'february': 28, 'march': 31, 'april': 30, 'may': 31, 'june': 30,
            'july': 31, 'august': 31, 'september': 30, 'october': 31, 'november': 30, 'december':31}
    return year[month]
def order_list(num):
    length = len(num)
    for i in range(length):
        for j in range(length - i - 1):
            if num[j] > num[j + 1]:
                num[j], num[j + 1] = num[j + 1], num[j]
    return(num)
# Numpy
import numpy as np
# def division():
#     a = np.arange(1, 1001)
#     return np.array([i for i in a if i % 5 == 0 or i % 7 == 0])
# def division():
#     a = np.arange(2, 1002)
#     b = (a % 5 == 0) + (a % 7 == 0)

#     return np.where(b)
# print(division())
# def sort(num):
#     return np.sort(num)

print(True + False and False)
print(False + False)
prin#
