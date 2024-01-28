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


# 定义模型结构和参数
def create_model():
    model = Sequential()
    model.add(LSTM(units=5, return_sequences=True, input_shape=(features_scaled.shape[1], 1)))
    model.add(LSTM(units=5))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 初始化交叉验证和MSE
kfolds = KFold(n_splits=5, shuffle=True)
mse_scores = []

# 进行交叉验证
for train_idx, test_idx in kfolds.split(features_scaled):
    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]

    # 将数据转换为LSTM所需的三维输入格式 [样本数，时间步长，特征数]
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 创建并训练模型
    model = create_model()
    model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, verbose=0)

    # 使用测试集进行预测
    predictions = model.predict(X_test_lstm)

    # 计算MSE
    mse = np.mean((predictions - y_test)**2)
    mse_scores.append(mse)

model.save('trained_LSTMmodel.h5')

# 输出模型的MSE
mean_mse = np.mean(mse_scores)
print("Mean Squared Error (MSE):", mean_mse)