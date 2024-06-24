#%%
import pandas as pd
import numpy as np
# #%%
# dfstocks = pd.read_csv('./data/CSI.csv')
# dfstocks

# # %%
# pd.read_csv('./data/aapl_call0707_0714.csv')
# # %%
# df = pd.DataFrame(index = [1, 2, 3, 4, 5], columns = ['A', 'B', 'C', 'D'], 
#                   data = [['a', 'b', 'c', 'd'],
#                           ['aa', 'bb', 'cc', 'dd'],
#                           ['aaa', 'bbb', 'ccc', 'ddd'],
#                           ['aaaa', 'bbbb', 'cccc', 'dddd'],
#                           ['aaaaa', 'bbbbb', 'ccccc', 'ddddd']])
# print(df)

# row2 = df.loc[2]
# row2_num = df.iloc[1]
# print(row2, row2_num)

# element = df.at[3, 'A']
# print(element)

#%%
# from datetime import date, datetime, time, timedelta
# mydate_now = date.today()
# mydatetime_now = datetime.now()
# print(f'mydate_now:{mydate_now}')
# print(f'mydatetime_now{mydatetime_now}')

# mydate = date(2024, 2, 28) # you cannot fill a date that is unreal
# print(f'Specifiied date:{mydate}, with type:{type(mydate)}')
# mytime = time(0, 30, 59)
# print(f'Specified time:{mytime}, with type:{type(mytime)}')
# mydatetime = datetime(2023, 2, 28, 00, 39, 51)
# print(f'Specified datetime:{mydatetime}, with type:{type(mydatetime)}')
# print(f'year:{mydatetime.year}, month:{mydatetime.month}, day:{mydatetime.day}, hour:{mydatetime.hour}, minute:{mydatetime.minute}, second:{mydatetime.second}')
# mydatetimestring = mydatetime.strftime('%Y-%m-%d %H:%M:%S')
# print(f'Formatted datetime: {mydatetimestring}, with type: {type(mydatetimestring)}')
# mydatetimestring = mydatetime.strftime('%A %d %B %Y')
# print(f'Formatted datetime: {mydatetimestring}, with type: {type(mydatetimestring)}')
# mydateasstring = '2024-02-29 14:30:59'
# mydatetime = datetime.strptime(mydateasstring, '%Y-%m-%d %H:%M:%S')
# print(f'Formatted datetime: {mydatetime}, with type: {type(mydatetime)}')

# mytime1 = datetime.strptime('2008-04-16 16:30:59', '%Y-%m-%d %H:%M:%S')
# mytime2 = datetime.strptime('2005-02-28 14:29:37', '%Y-%m-%d %H:%M:%S')
# delta1 = mytime1 - mytime2
# delta2 = mytime2 - mytime1
# print(f'time difference between {mytime1} and {mytime2} is {delta1}')
# print(f'time difference between {mytime2} and {mytime1} is {delta2}')
# print(f'the sum of deltas is {delta1+delta2}')
# print(mytime1.timestamp())
# mytime = datetime.fromtimestamp(3000000000)
# print(mytime)

# #%%
# data = np.array([[1, 2, 3, 4],
#                  [11, 22, 33, 44],
#                  [111, 222, 333, 444],
#                  [1111, 2222, 3333, 4444]])
# df = pd.DataFrame(columns = ['A', 'B', 'C', 'D'], index = [1, 2, 3, 4], data = data)
# a = df.mean(axis = 0)
# b = df.mean(axis = 1)
# print(f'a\n{a}')
# print(f'b\n{b}')

# aa = df.std(axis = 0)
# bb = df.std(axis = 1)
# print(f'aa\n{aa}')
# print(f'bb\n{bb}')

#%%
# N = 20
# r1, r2, r3, r4 = np.ones(N), np.ones(N), np.ones(N), np.ones(N)
# # print(r1, r2, r3, r4)
# for i in range(1, N):
#     r1[i] = 1
#     r2[i] = r2[i - 1] + 1
#     r3[i] = r3[i - 1] * 2
# for i in range(2, N):
#     r4[i] = r4[i - 1] + r4[i - 2]
# # print(r1, r2, r3, r4)
# data1 = np.concatenate((r1, r2))
# data2 = np.concatenate((r4, r3))
# data1 = np.reshape(data1, (2, N))
# data2 = np.reshape(data2, (2, N))
# # print(data1, data2)
# data = np.hstack((data1.T, data2.T))
# # print(data)
# se = pd.DataFrame(index = np.arange(N), columns = ['A', 'B', 'C', 'D'], data = data)
# # print(se)

# ses = se.shift(1).fillna(1)
# # print((se - ses) / ses) # = derivative
# # print(se.cumsum()) # = Sigma(se)
# # print(se.cumprod()) # = n!

# # so = se[se['C'] >= 10][se['B'] < 10]
# # print(so)

#%%

data1 = np.array([['P', 'P', 'P', 'M', 'M', 'M']]).T
data2 = np.array([['a', 'b', 'c', 'a', 'b', 'c']]).T
data3 = np.array([[94, 85, 99, 82, 87, 90]]).T
data4 = np.array([[2, 3, 1, 3, 2, 1]]).T
sub = pd.DataFrame(columns = ['subject'], data = data1)
stu = pd.DataFrame(columns = ['student'], data = data2)
mrk = pd.DataFrame(columns = ['marking'], data = data3)
rnk = pd.DataFrame(columns = ['ranking'], data = data4)

te = pd.concat([sub, stu, mrk, rnk], axis = 1)
print(te)
print(te['marking'].dtype)

tsub = te.groupby('subject')['marking'].mean()
print(tsub)
tstu = te.groupby('student')['marking'].mean()
print(tstu)
tss = te.groupby(['subject', 'student'])['marking'].mean()
print(tss)

# def adjust(X):
    # return np.max(X)
# def give_rankings(X):
    # return np.sum(X)

# ad = te.groupby('student').agg(Markings = ('marking', 'max'), Rankings = ('ranking', 'min'))
# print(ad)

# construct a historical scene:
# you can travel to any time in the history, for only once
# we have two conditions: 
# 1.time and location (specific or random) 
# 2.body mind stuff (unchange or change)
# specific time and location makes this easier, and these become our hypothesis
# if bring your body, there are many problems:
#       1. disease
#           a. the condition of medical is quite bad
#           b. you may bring a lot of diseases and cause a plague
#       2. appearance
#           a. people are usually skinny and short and tanned
#       3. language
#           a. spoken-language is far differnent from today
#           b. characters are different
#           c. the way they speak is different 
#       4. habbit
#           a. mind and habbit are ridiculous seen from before     
#       5. condition
#           a. food, water, clothes, labor, environment, pleasure
# then we decide not to bring our body, but travel to someone's body
# to solve language problems, we assume that we have inherited all the things they have
# now we know the the usage of language before, and we have all the memory from them
# in that circumstance, we just think that we wake up from a dream of far away future
# you remember all the knowledge and memory and sense
# now let's choose our background:
#       1. time
#           Tang Dynastic (618 - 907)(suggested: 618 - 755)
#       2. location
#           Changan(capital)
#       3. body
#           Man
#       4. professional
#           bartender of a midium bar
#       5. stage
#           normal citizen

# some details:
# mirrors, toothbrush, waterbottle, lamp, clock, hair, clothes, building, street, vihiecle
# drums for time, curfew, cock, thatch bed, hard pillow, bare foot/ thatch shoes, 
# officer, the rich, horse, cars, pedestrain, road
# charcoal, coal, carter, provisons and forage, cotton, silk, wood
# guard, soilder
# tree, sand, well, water, building, wall
# customer, poet, examinee
# tax
# breakfast, lunch, dinner
# sky, tempreture, sun, night, star, moon
# job, hobby