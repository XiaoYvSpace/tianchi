import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 读取csv文件
# data = pd.read_csv('pearsonr_top_100_features.csv')
data = pd.read_csv('boruta_selected_features.csv')


# 提取特征列和目标列
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将特征和目标转换为numpy数组
X = np.array(X)
y = np.array(y)

max_vals = np.max(X, axis=0)
min_vals = np.min(X, axis=0)

# 归一化每列的值到0到1之间
X = (X - min_vals) / (max_vals - min_vals)

# 定义K折交叉验证
kfold = KFold(n_splits=5)

# 初始化变量用于记录每个折的MSE
mse_scores = []

# 创建卷积神经网络模型
def create_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model


# 循环进行交叉验证
for train_index, test_index in kfold.split(X):
    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 将输入数据整形为卷积层期望的形状
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # 创建模型
    model = create_model()

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # 预测测试集
    y_pred = model.predict(X_test)
    # print("预测值：",y_pred)

    # 计算MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    # mse = np.mean((y_pred - y_test)**2)
    # mse_scores.append(mse)

model.save('trained_CNNmodel.h5')

# 计算平均MSE
mean_mse = np.mean(mse_scores)
print("均方误差（MSE）：", mean_mse)