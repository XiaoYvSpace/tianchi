import pandas as pd
import numpy as np

# 读取csv文件
data = pd.read_csv('output_fill.csv')

# 删除常数值列
data = data.loc[:, (data != data.iloc[0]).any()]

# 删除方差小于0.00001的列
data = data.loc[:, data.var() >= 0.00001]

X = data.iloc[:, :-1]  # 替换为你的特征列
y = data['Value']  # 替换为你的目标变量列

# 使用Spearman等级相关性计算相关性矩阵
correlation_matrix = X.corrwith(y, method='spearman')

# 获取特征之间的相关性排序
sorted_correlation = correlation_matrix.sort_values(ascending=False)

# 获取排名前100名的列名和相关性
#top_100_features = sorted_correlation[:100]
top_100_features = sorted_correlation[:50]
top_100_features = top_100_features[~top_100_features.index.duplicated()]  # 去除重复的索引

# 获取排名前100名的列名
top_100_features_names = top_100_features.index.tolist()
top_100_features_names.append('Value')  # 将 "Value" 列名加入列表

# 获取列名和取值的字典
top_100_features_values = data[top_100_features_names].to_dict(orient='list')

# 创建包含列名和取值的DataFrame
top_100_df = pd.DataFrame(top_100_features_values, columns=top_100_features_names)

# 将排名前100名的列名和取值保存为新的csv文件
top_100_df.to_csv('spearman_top_100_features.csv', index=False)

