from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

# 加载模型
model = load_model('trained_CNNmodel.h5')

# 读取测试集CSV文件
Val_features = pd.read_csv('boruta_selected_testA_fill.csv')
Val_target = pd.read_csv('dataset_answerA.csv')

# 分割特征和标签
val_features = Val_features.iloc[1:, :]
val_target = Val_target.iloc[:, -1] 

max_vals = np.max(val_features, axis=0)
min_vals = np.min(val_features, axis=0)

# 归一化每列的值到0到1之间
val_features = (val_features - min_vals) / (max_vals - min_vals)

# 提取特征列
X_to_predict = np.array(val_features)

# 将输入数据整形为卷积层期望的形状
X_to_predict = X_to_predict.reshape(X_to_predict.shape[0], X_to_predict.shape[1], 1)

# 进行预测
val_predictions = model.predict(X_to_predict)

mse = mean_squared_error(val_target, val_predictions)
print("Validation Mean Squared Error (MSE):", mse)