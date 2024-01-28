import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

def model_adjust_parameters(cv_params, other_params):
    """模型调参：GridSearchCV"""
    # 模型基本参数
    model = xgb.XGBRegressor(**other_params)
    # sklearn提供的调参工具，训练集k折交叉验证(消除数据切分产生数据分布不均匀的影响)
    # optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1)
    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1)
    # 模型训练
    optimized_param.fit(X, y)
     # 获取最佳模型
    best_model = optimized_param.best_estimator_
    # 保存模型
    best_model.save_model('trained_XGBmodel.model')
    # 对应参数的k折交叉验证平均得分
    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_score: %f,  params: %r" % (mean, param))
    # 最佳模型参数
    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
    # 最佳参数模型得分
    print('最佳模型得分:{0}'.format(optimized_param.best_score_))
    


# 读取CSV文件
data = pd.read_csv('output_fill.csv')

# 分割特征和标签
X = data.iloc[:, :-1]  # 所有行，除了最后一列
y = data.iloc[:, -1]  # 最后一列

# 进行特征缩放（与训练过程中使用的缩放器相同）
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 定义模型参数的取值范围
cv_params = {
    'n_estimators': [1200,1800,2100],
    'max_depth': [1,2,3, 5],
    'min_child_weight':[3,4,5,7,9,11],
    'learning_rate': [ 0.1,0.01]
}

# 定义模型的其他参数
other_params = {
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# 调用模型调参函数
model_adjust_parameters(cv_params, other_params)

