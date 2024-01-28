import pandas as pd

# 读取第一个csv文件
df1 = pd.read_csv('converted_file_encoded.csv')

# 读取第二个csv文件
df2 = pd.read_csv('testA_converted.csv')

# 在df2中添加encoder_test列，默认值为NaN
df2['encoder_test'] = float('nan')

# 逐行遍历第一个csv文件，将encoder列的值赋给df2对应行的encoder_test列
for index, row in df1.iterrows():
    mask = (df2.iloc[:, :11] == row[:11]).all(axis=1)  # 创建用于匹配的布尔掩码
    df2.loc[mask, 'encoder_test'] = row['encoder']

# 保存结果为新的csv文件
df2.to_csv('testA_converted_encoded.csv', index=False)

# 找出df2中encoder_test为NaN的行，即跟第一个文件所有行都不相同的行
non_duplicate_rows = df2[df2['encoder_test'].isna()]

# 打印不相同的行和行名
print(non_duplicate_rows)