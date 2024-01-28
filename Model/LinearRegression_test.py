import pandas as pd
import h5py
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
#Val_features = pd.read_csv('boruta_selected_testA_fill.csv')
Val_target = pd.read_csv('dataset_answerA.csv')

data1 = pd.read_csv('boruta_selected_testA_fill.csv')
#data2=pd.read_csv('testA_converted.csv')
#Val_features = pd.concat([data2, data1], axis=1)

# 分割特征和标签
#features = data1.iloc[1:, :]
features = data1.iloc[1:, :]
target = Val_target.iloc[:, -1] 

# 加载保存的模型.h5文件
with h5py.File('trained_transformer.h5', 'r') as file:
    weights = file['model_weights'][:]
    intercept = file['intercept'][()]

# 构建模型
regressor = LinearRegression()
regressor.coef_ = weights
regressor.intercept_ = intercept

# 使用模型进行预测
predictions = regressor.predict(features)

# 打印预测结果
print(predictions)

# 计算MSE并输出
mse = mean_squared_error(target, predictions)
print("MSE:", mse)