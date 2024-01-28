import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import h5py
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
data1 = pd.read_csv('boruta_selected_features.csv')
# data2=pd.read_csv('converted_file.csv')
# data = pd.concat([data2, data1], axis=1)

#data=pd.read_csv('boruta_selected_features.csv')

# 获取特征和目标变量
features = data1.iloc[:, :-1]
target = data1.iloc[:, -1]

# 创建线性回归模型
regressor = LinearRegression()

# 创建Transformer来处理目标变量
transformer = TransformedTargetRegressor(regressor=regressor)

# 进行五折交叉验证训练并输出MSE
#mse_scores = -cross_val_score(transformer, features, target, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cross_val_score(transformer, features, target, cv=5, scoring='neg_mean_squared_error', error_score='raise')
print("MSE scores:", mse_scores)
print("Average MSE:", np.mean(mse_scores))

# 使用完整数据集训练模型
transformer.fit(features, target)

# 保存训练好的模型为.h5文件
with h5py.File('trained_transformer.h5', 'w') as file:
    file.create_dataset('model_weights', data=transformer.regressor_.coef_)
    file.create_dataset('intercept', data=transformer.regressor_.intercept_)

