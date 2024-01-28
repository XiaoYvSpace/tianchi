import pandas as pd
import numpy as np

# 读取第一个CSV文件
data1 = pd.read_csv('output.csv')
data1=data1.iloc[:, -11:]

# 读取第二个CSV文件
data2 = pd.read_csv('testA_process.csv')
data2=data2.iloc[:,:]

# # 获取需要转换的列名
# common_columns = list(set(data1.columns).intersection(data2.columns))
# print(common_columns)

# # 合并两个文件中相同列名的数据
# merged_data = pd.concat([data1[common_columns], data2[common_columns]])

# 拼接两个数据集
merged_data = pd.concat([data1, data2])

# 对每一列进行字母到数字的转换
for column in merged_data:
    unique_letters = merged_data[column].unique()
    unique_letters.sort()
    letter_to_number = {letter: number for number, letter in enumerate(unique_letters)}
    merged_data[column] = merged_data[column].replace(letter_to_number)

# 将转换结果保存到各自的CSV文件
data1_converted = merged_data.head(len(data1))
data2_converted = merged_data.tail(len(data2))
data1_converted.to_csv('converted_file_sort.csv', index=False)
data2_converted.to_csv('testA_converted.csv', index=False)