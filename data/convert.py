import pandas as pd
import numpy as np

# 读取CSV文件
data = pd.read_csv('testA_process.csv')

# 获取最后11列的数据
last_11_columns = data.iloc[:, :]
print(last_11_columns)

# 对每一列进行字母到数字的转换
for column in last_11_columns:
    unique_letters = last_11_columns[column].unique()
    unique_letters.sort()  # 对unique_letters进行排序
    letter_to_number = {letter: number for number, letter in enumerate(unique_letters)}
    last_11_columns[column] = last_11_columns[column].replace(letter_to_number)

# 将转换结果保存到新的CSV文件
last_11_columns.to_csv('testA_converted.csv', index=False)

# import pandas as pd
# import numpy as np

# # 读取CSV文件
# data = pd.read_csv('output.csv')

# # 获取最后11列的数据
# last_11_columns = data.iloc[:, -11:]
# print(last_11_columns)

# # 对每一列进行字母到数字的转换
# for column in last_11_columns:
#     unique_letters = last_11_columns[column].unique()
#     unique_letters.sort()  # 对unique_letters进行排序
#     letter_to_number = {letter: number for number, letter in enumerate(unique_letters)}
#     last_11_columns[column] = last_11_columns[column].replace(letter_to_number)

# # 将转换结果保存到新的CSV文件
# last_11_columns.to_csv('converted_file_sort.csv', index=False)