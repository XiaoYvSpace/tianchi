import pandas as pd
from scipy.stats import pearsonr

# 读取csv文件
data = pd.read_csv('output_fill.csv')

# 计算特征之间的相关系数
correlation_matrix = data.corr()

# 计算每个特征与目标变量之间的相关系数
target_correlation = {}
target_variable = 'Value'  # 替换为你的目标变量名称

for column in data.columns:
    if column != target_variable:
        is_constant = (data[column].std() == 0)
        is_near_constant = (data[column].std() < 1e-5 * (data[column].max() - data[column].min()))
        if not is_constant and not is_near_constant:
            correlation = pearsonr(data[column], data[target_variable])[0]
            target_correlation[column] = correlation

# 按照相关系数的绝对值进行排序
sorted_correlation = sorted(target_correlation.items(), key=lambda x: abs(x[1]), reverse=True)

# 获取排名前100名的列名和取值
#top_100_features = sorted_correlation[:100]
top_100_features = sorted_correlation[:50]
top_100_features_names = [x[0] for x in top_100_features]
top_100_features_names.append(target_variable)  # 将 "Value" 列名加入列表

# 从csv文件中获取对应列的所有取值
top_100_features_values = {}

for feature, _ in top_100_features:
    top_100_features_values[feature] = data[feature].values.tolist()

# 添加最后一列 "Value" 的取值
top_100_features_values[target_variable] = data[target_variable].values.tolist()

# 创建包含列名和取值的DataFrame
top_100_df = pd.DataFrame(top_100_features_values, columns=top_100_features_names)

# 将排名前100名的列名和取值保存为新的csv文件
top_100_df.to_csv('pearsonr_top_100_features.csv', index=False)