import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape , Concatenate,Dense
from keras.losses import mean_squared_error
import tensorflow as tf

# 加载CSV文件
data = pd.read_csv('testA_selected_same_columns.csv')

# 提取特征列
data = data.iloc[:, :]

# 将特征和目标转换为numpy数组
data = np.array(data)

max_vals = np.max(data, axis=0)
min_vals = np.min(data, axis=0)

# 检查是否存在 max_vals = min_vals 的列
delete_cols = []
for i in range(len(max_vals)):
    if max_vals[i] == min_vals[i]:
        delete_cols.append(i)
        print(i)

# 删除 max_vals 和 min_vals 中对应的值
max_vals = np.delete(max_vals, delete_cols)
min_vals = np.delete(min_vals, delete_cols)

# 删除 max_vals = min_vals 的列
if delete_cols:
    data = np.delete(data, delete_cols, axis=1)

# 归一化每列的值到 0 到 1 之间
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# 获取 data 的列数
num_columns = data.shape[1]
print("data列数：",num_columns)

# 创建一个全为 1 的列向量
ones_column = np.ones((data.shape[0], 3))

# 将全为 1 的列向量添加到 data 的最后一列
data = np.concatenate((data, ones_column), axis=1)

T=300
W=2620
#encoding_dim=(T*W)/(3*20*67)-1
encoding_dim=3
x = np.reshape(data, (1, T, W, 1))

x_input = Input(shape=(T, W, 1))   
conv1_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x_input)    
pool1 = MaxPooling2D((5, 5), padding='same')(conv1_1)   
conv1_2 = Conv2D(4, (3, 3), activation='relu', padding='same')(pool1)  
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)   
conv1_3 = Conv2D(2, (3, 3), activation='relu', padding='same')(pool2)  
h = MaxPooling2D((2, 2), padding='same')(conv1_3)
encoded = Dense(encoding_dim, activation='relu')(h)
encoded_Flatten = Flatten()(encoded)

input_shape = h.shape[-1:]
input_shape = np.prod(input_shape)

up0 = Dense(input_shape, activation='relu')(encoded)
conv2_1 = Conv2D(2, (3, 3), activation='relu', padding='same')(up0)  
up1 = UpSampling2D((2, 2))(conv2_1) 
conv2_2 = Conv2D(4, (3, 3), activation='relu', padding='same')(up1)    
up2 = UpSampling2D((2, 2))(conv2_2) 
conv2_3 = Conv2D(8, (3, 3), activation='relu',padding='same')(up2)    
up3 = UpSampling2D((5, 5))(conv2_3) 
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)


autoencoder = Model(inputs=x_input, outputs=r)  
autoencoder.compile(optimizer='adam', loss=mean_squared_error)

# 输出降维后的表示维度
print(x.shape)
print(pool1.shape)
print(pool2.shape)
print(encoded.shape)

# 加入训练模块进行模型训练
autoencoder.fit(x, x, epochs=50, batch_size=32)

loss = autoencoder.history.history['loss']
print(loss)

# 获取降维后的结果
encoder = Model(inputs=x_input, outputs=encoded_Flatten)
encoded_data = encoder.predict(x)

# 将降维结果保存为CSV文件
encoded_df = pd.DataFrame(encoded_data)
encoded_df.to_csv('testA_encoded_others_data.csv', index=False)