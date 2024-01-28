import pandas as pd

# 读取csv文件
data = pd.read_csv('output_fill.csv')

# 获取除最后一列以外的所有列
X = data.iloc[:, :-1]

# 获取最后一列
y = data.iloc[:, -1]

# 使用Kendall等级相关性计算相关性矩阵
correlation_matrix = X.corrwith(y, method='kendall')

# 获取特征之间的相关性排序
sorted_correlation = correlation_matrix.sort_values(ascending=False)

# # 输出相关性最大的前200个特征
# top_200_features = sorted_correlation[:200]

# for feature, correlation in top_200_features.items():
#     print(f"{feature}: {correlation}")

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
top_100_df.to_csv('kendall_top_100_features.csv', index=False)