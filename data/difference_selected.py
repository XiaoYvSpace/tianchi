import pandas as pd

# 读取第一个csv文件
df1 = pd.read_csv('dataset_testA.csv')

num_columns = df1.shape[1]

# 输出列数
print("Columns:", num_columns)

# 读取第二个csv文件
df2 = pd.read_csv('boruta_selected_testA.csv')

num_columns_2 = df2.shape[1]

# 输出列数
print("Columns:", num_columns_2)

# 获取第一个csv文件的列名
columns1 = df1.columns

# 获取第二个csv文件的列名
columns2 = df2.columns
print(columns2)

# 获取第一个csv文件中不在第二个csv文件中的列名
columns_to_keep = [col for col in columns1 if col not in columns2]

# 从第一个csv文件中选择要保留的列
new_df = df1[columns_to_keep]

# 将结果保存为新的csv文件
new_df.to_csv('testA_others_selected.csv', index=False)