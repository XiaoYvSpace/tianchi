import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import h5py

data_1 = pd.read_csv('converted_file_encoded.csv')
data_1 = data_1.iloc[:, -1]
max_vals = np.max(data_1, axis=0)
min_vals = np.min(data_1, axis=0)
# 归一化每列的值到0到1之间
data_1 = (data_1 - min_vals) / (max_vals - min_vals)


data_2_pre = pd.read_csv('boruta_selected_features.csv')
data_2 = data_2_pre.iloc[:, :-1]
y = data_2_pre.iloc[:, -1]
max_vals = np.max(data_2, axis=0)
min_vals = np.min(data_2, axis=0)
# 归一化每列的值到0到1之间
data_2 = (data_2 - min_vals) / (max_vals - min_vals)

# 读取其他工序数据文件
data_3 = pd.read_csv('encoded_others_data.csv', usecols=range(16000))  # 将路径替换为你的数据文件路径

# 将数据重构为800行887列的数组
reshaped_data = np.array(data_3).reshape((800, 20))
max_vals = np.max(reshaped_data, axis=0)
min_vals = np.min(reshaped_data , axis=0)
# 归一化每列的值到0到1之间
reshaped_data  = (reshaped_data  - min_vals) / (max_vals - min_vals)

data = pd.concat([data_1, 200*data_2, pd.DataFrame(reshaped_data)], axis=1)

features = data.iloc[:, :]
target = y

features.columns = features.columns.astype(str)

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
with h5py.File('trained_all_transformer.h5', 'w') as file:
    file.create_dataset('model_weights', data=transformer.regressor_.coef_)
    file.create_dataset('intercept', data=transformer.regressor_.intercept_)