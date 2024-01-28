import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error

# 导入训练好的模型
model = load_model('trained_LSTMmodel.h5')

# # 读取验证集CSV文件
# val_data = pd.read_csv('boruta_selected_testA.csv')

# # 提取特征和目标变量
# val_features = val_data.iloc[:, :-1].values
# val_target = val_data.iloc[:, -1].values

# 读取测试集CSV文件
Val_features = pd.read_csv('pearsonr_selected_testA_fill.csv')
Val_target = pd.read_csv('dataset_answerA.csv')

# 分割特征和标签
val_features = Val_features.iloc[1:, :]
val_target = Val_target.iloc[:, -1] 

# 进行特征缩放（与训练过程中使用的缩放器相同）
scaler = MinMaxScaler()
val_features_scaled = scaler.fit_transform(val_features)

# 将验证集的数据转换为LSTM所需的三维输入格式 [样本数，时间步长，特征数]
val_features_lstm = np.reshape(val_features_scaled, (val_features_scaled.shape[0], val_features_scaled.shape[1], 1))

# 进行验证集的预测
val_predictions = model.predict(val_features_lstm)

# val_predictions = np.squeeze(val_predictions)
# val_target = np.squeeze(val_target)

# 计算验证集MSE
#val_mse = np.mean((val_predictions - val_target)**2)
mse = mean_squared_error(val_target, val_predictions)
print("Validation Mean Squared Error (MSE):", mse)