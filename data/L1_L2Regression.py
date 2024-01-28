import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

# 读取CSV文件
data = pd.read_csv("output_fill.csv")

# 分离特征和标签
X = data.iloc[:, :-1]  # 替换为你的特征列
y = data['Value']  # 替换为你的目标变量列

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建LASSO回归模型对象
lasso = LassoCV(cv=5, max_iter=10000)

# 拟合模型并进行特征选择
lasso.fit(X, y)

# 输出特征选择结果
selected_features_lasso = X.columns[lasso.coef_ != 0]
print("lasso Selected features:", selected_features_lasso)

# 生成新的CSV文件
selected_data = data[selected_features_lasso]
selected_data.to_csv("lasso_selected_features.csv", index=False)

# # 创建带有L1正则化的逻辑回归模型对象
# logreg = LogisticRegression(penalty='l1')

# # 拟合模型并进行特征选择
# logreg.fit(X, y)

# # 输出特征选择结果
# selected_features_L1 = X.columns[logreg.coef_[0] != 0]
# print("L1 Selected features:", selected_features_L1)

# # 创建带有L2正则化的逻辑回归模型对象
# logreg = LogisticRegression(penalty='l2')

# # 拟合模型并进行特征选择
# logreg.fit(X, y)

# # 输出特征选择结果
# selected_features_L2 = X.columns[logreg.coef_[0] != 0]
# print("L2 Selected features:", selected_features_L2)