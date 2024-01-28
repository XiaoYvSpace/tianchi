from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# 加载训练好的模型
model = load_model('trained_FCmodel.h5')

# 读取测试集CSV文件
# Val_features = pd.read_csv('boruta_selected_testA_fill.csv')
Val_features = pd.read_csv('pearsonr_selected_testA_fill.csv')
Val_target = pd.read_csv('dataset_answerA.csv')

# 分割特征和标签
val_features = Val_features.iloc[1:, :]
val_target = Val_target.iloc[:, -1] 

max_vals = np.max(val_features, axis=0)
min_vals = np.min(val_features, axis=0)

# 归一化每列的值到0到1之间
val_features = (val_features - min_vals) / (max_vals - min_vals)

# 提取特征列
X_test = np.array(val_features)

# 使用模型进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(val_target, y_pred)

print('均方误差(MSE):', mse)