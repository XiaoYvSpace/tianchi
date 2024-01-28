
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 读取CSV文件
data = pd.read_csv('pearsonr_top_100_features.csv')

# 提取特征和目标变量
features = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# 数据归一化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 初始化交叉验证和MSE
kfolds = KFold(n_splits=5, shuffle=True)
mse_scores = []
best_params = {}
best_mse = float('inf')

# 定义可选参数
# unit_options = [10, 25, 50, 75, 100]
# time_steps_options = [1, 3, 5]
# epochs_options = [50, 100]
# batch_sizes_options = [32, 64]
unit_options = [2,3,4,5]
time_steps_options = [5,10]
epochs_options = [25,50]
batch_sizes_options = [64]

# 进行参数调整和交叉验证
for units in unit_options:
    for time_steps in time_steps_options:
        for epochs in epochs_options:
            for batch_size in batch_sizes_options:
                mse_scores = []

                # 进行交叉验证
                for train_idx, test_idx in kfolds.split(features_scaled):
                    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
                    y_train, y_test = target[train_idx], target[test_idx]

                    # 将数据转换为LSTM所需的三维输入格式 [样本数，时间步长，特征数]
                    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                    # 创建并训练模型
                    model = Sequential()
                    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
                    model.add(LSTM(units=units))
                    model.add(Dense(units=1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                    # 使用测试集进行预测
                    predictions = model.predict(X_test_lstm)

                    # 计算MSE
                    mse = np.mean((predictions - y_test)**2)
                    mse_scores.append(mse)

                # 计算平均MSE
                mean_mse = np.mean(mse_scores)

                # 更新最佳参数和最佳MSE
                if mean_mse < best_mse:
                    best_mse = mean_mse
                    best_params = {'units': units, 'time_steps': time_steps, 'epochs': epochs, 'batch_size': batch_size}

# 输出最佳参数和最佳MSE
print("Best Parameters:", best_params)
print("Best Mean Squared Error (MSE):", best_mse)