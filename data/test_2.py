import pandas as pd

# 读取第一个csv文件
df1 = pd.read_csv('others_selected.csv')

num_columns = df1.shape[1]

# 输出列数
print("Columns:", num_columns)

# 读取第二个csv文件
df2 = pd.read_csv('dataset_select_testA_fill.csv')

num_columns_2 = df2.shape[1]

# 输出列数
print("Columns:", num_columns_2)

# 获取第一个csv文件的列名
columns1 = df1.columns

# 获取第二个csv文件的列名
columns2 = df2.columns
#print(columns2)

# 获取第二个csv文件中和第一个csv文件列名相同的列名
columns_to_keep = [col for col in columns2 if col in columns1]
print("保留列长度：",len(columns_to_keep))

# 从第二个csv文件中选择要保留的列
new_df = df2[columns_to_keep]


# 将结果保存为新的csv文件
new_df.to_csv('testA_selected_same_columns.csv', index=False)