import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
# data = pd.read_csv('boruta_selected_features.csv')
data = pd.read_csv('pearsonr_top_100_features.csv')

# 获取特征和目标变量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(y)

# # 标准化特征
# mean = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# X = (X - mean) / std

# 进行特征缩放(归一化)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 定义五折交叉验证
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# 定义保存MSE的列表
mse_scores = []

# 进行五折交叉验证
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 创建模型
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # 编译模型
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)

    # 进行预测
    y_pred = model.predict(X_test)

    # 计算MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

model.save('trained_FCmodel.h5')

# 打印五折交叉验证的MSE平均值
print("Mean MSE:", np.mean(mse_scores))