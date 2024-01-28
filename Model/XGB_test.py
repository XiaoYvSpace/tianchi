import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 调用模型调参函数
#model = xgb.XGBRegressor(learning_rate=0.01, max_depth=2, min_child_weight=9, n_estimators=1800)

model = xgb.Booster()
model.load_model('trained_XGBmodel.model')

# 读取测试集CSV文件
test_data = pd.read_csv('lasso_selected_testA.csv')
answer_data = pd.read_csv('dataset_answerA.csv')

# 分割特征和标签
X_test = test_data.iloc[1:, :]  
y_test = answer_data.iloc[:, -1]  

#使用模型预测
# 将X_test转换为DMatrix对象
dtest = xgb.DMatrix(X_test)

# 然后使用转换后的dtest进行预测
y_pred = model.predict(dtest)
#y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)

print("均方误差（MSE）:", mse)



