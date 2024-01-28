import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 读取CSV文件
data = pd.read_csv('converted_file.csv')
input_data = data.values

# 定义输入和输出维度
input_dim = input_data.shape[1]
output_dim = 1

# 定义自编码器的参数
hidden_dim1 = 32
hidden_dim2 = 4
hidden_dim3 = 1

# 构建自编码器的计算图
input_layer = Input(shape=(input_dim,))
encoded_1 = Dense(hidden_dim1, activation='relu')(input_layer)
encoded_2 = Dense(hidden_dim2, activation='relu')(encoded_1)
encoded_3 = Dense(hidden_dim3, activation='relu')(encoded_2)

decoded_1 = Dense(hidden_dim1, activation='relu')(encoded_2)
decoded_2 = Dense(hidden_dim1, activation='relu')(encoded_1)
decoded_3 = Dense(input_dim, activation='sigmoid')(decoded_2)

autoencoder = Model(input_layer, decoded_3)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# 训练自编码器
num_epochs = 100
batch_size = 32

history = autoencoder.fit(input_data, input_data, 
                          epochs=num_epochs, batch_size=batch_size)

# 提取降维后的特征
encoder = Model(inputs=input_layer, outputs=encoded_3)
encoded_output = encoder.predict(input_data)

# 输出降维后的特征
print("Encoded data shape:", encoded_output.shape)

# 将降维结果保存为CSV文件
encoded_df = pd.DataFrame(encoded_output)
encoded_df.to_csv('encoded_gongxu_data.csv', index=False)

# 输出损失值
loss_values = history.history['loss']
print("Loss:", loss_values)