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
