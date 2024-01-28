import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

# 读取csv文件
data = pd.read_csv('output_fill.csv')

# 将特征和目标变量分开
X = data.iloc[:, :-1]  # 替换为你的特征列
y = data['Value']  # 替换为你的目标变量列

# 初始化随机森林回归器
rf = RandomForestRegressor()

# 初始化Boruta特征选择器
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2)

# 运行Boruta特征选择
boruta_selector.fit(X.values, y.values)

# 获取选择的特征排名
feature_ranks = boruta_selector.ranking_

# 根据排名找出与目标列最相关的特征
#selected_features = X.columns[feature_ranks <= 200]

# 获取被确认和有潜在的特征列
selected_features = X.columns[boruta_selector.support_]

# 创建包含选定特征的新数据集
selected_data = X[selected_features]

# 将结果保存到CSV文件
selected_data.to_csv('boruta_selected_features.csv', index=False)

print(f"Selected features: {selected_features}")
