import pandas as pd

# 读取数据
data = pd.read_csv("converted_file_sort.csv")
data_copy=data

# 初始化一个列表来记录相同的行名
same_rows = []

# 循环直到数据为空
while len(data) > 0:
    # 取出当前首行的行名
    a = data.iloc[0]

    # 找出与当前行完全相同的行名，并将其存储为数组
    indices = data.index[data.apply(lambda x: x.equals(a), axis=1)].tolist()

    # 将数组加入列表
    same_rows.append(indices)

    # 从数据中删除相同的行
    data = data.drop(indices)

# 输出相同的行名
print("Same rows:", same_rows)

# 获取same_rows中数组的个数
array_count = len(same_rows)

print(array_count)

# 构建编码与数组的对应关系
encode_mapping = {}
for i in range(array_count):
    for row_name in same_rows[i]:
        encode_mapping[row_name] = i

# 创建一个空的DataFrame
data_connection = pd.DataFrame()

# 设置行索引为0到799
data_connection.index = range(800)

# 验证行索引设置是否成功
print(data_connection.index) 

# 创建新列并填充编码值
data_connection['encoder'] = data_connection.index.map(encode_mapping.get)

print(data_connection)

new_data = pd.concat([data_copy, data_connection], axis=1)

# 将结果保存到新的csv文件
new_data.to_csv("converted_file_encoded.csv", index=False)


