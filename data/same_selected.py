import pandas as pd

# 读取第一个csv文件
df1 = pd.read_csv('converted_file.csv')

# 读取第二个csv文件
df2 = pd.read_csv('dataset_testA.csv')

# 获取第一个文件的列名
columns1 = df1.columns[:].tolist()

print(columns1)

# 获取第二个文件中和第一个文件列名相同的所有列
df2_selected = df2[columns1]

# 将选取的列保存为新的csv文件
df2_selected.to_csv('testA_process.csv', index=False)