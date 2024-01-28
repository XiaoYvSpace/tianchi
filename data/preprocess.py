import pandas as pd
import numpy as np


# 读取csv文件，跳过第一行表头，并不使用列头
# df = pd.read_csv('dataset_train.csv', skiprows=1, header=None)
df = pd.read_csv('dataset_train.csv',index_col=0)

# 获取包含字符串的列
string_columns = df.select_dtypes(include='object').columns
# 提取包含字符串的列并存为新的df
new_df = df[string_columns]

# 删除全空的列
df = df.dropna(axis=1, how='all')

# 删除包含字符串的列
df = df.drop(string_columns, axis=1)

# 删除方差恒为0的列
df = df.loc[:, df.var() != 0]

# 删除重复的列
df = df.T.drop_duplicates().T

# 将new_df和原df拼接成一个新的df
df = pd.concat([df, new_df], axis=1)

# 保存为新的csv文件
df.to_csv('output.csv', index=False)