import pandas as pd

# 读取第一个CSV文件
df1 = pd.read_csv('dataset_train.csv')

# 读取第二个CSV文件
df2 = pd.read_csv('output.csv')

# 获取两个文件的列名
columns1 = set(df1.columns)
columns2 = set(df2.columns)

# 获取两个文件的不同列
different_columns = list(columns1.symmetric_difference(columns2))

# 将不同列保存为新的CSV文件
df_diff = pd.DataFrame({'different_columns': different_columns})
df_diff.to_csv('different_columns.csv', index=False)