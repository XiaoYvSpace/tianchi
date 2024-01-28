import pandas as pd

# 读取CSV文件
df = pd.read_csv('dataset_select_testA.csv')

# #去掉最后11列
# df = df.iloc[:, :-11]

# # 删除空缺值较多的列
# columns_to_drop = []
# for column in df.columns:
#     if df[column].isnull().sum() > 400:
#         columns_to_drop.append(column)
# df = df.drop(columns=columns_to_drop)

# 计算每列的平均值
means = df.mean()

# 填补缺失值
df_filled = df.fillna(means)

# 保存为新的CSV文件
df_filled.to_csv('dataset_select_tastA_fill.csv', index=False)