import pandas as pd

# 读取csv文件
df = pd.read_csv('output.csv')

# 去除后11列
df = df.iloc[:, :-11]

# 获取每一列的平均值
mean_values = df.mean()

# 获取每一列的中位数
median_values = df.median()

# 获取每一列的最大值
max_values = df.max()

# 获取每一列的最小值
min_values = df.min()

# 获取每一列中为0的个数
zero_counts = df.eq(0).sum()

# 获取每一列中的空值个数
null_counts = df.isnull().sum()

# 获取除0和空值以外每一列的平均值
nonzero_mean_values = df[df != 0].mean()

# 获取除0和空值以外每一列的中位数
nonzero_median_values = df[df != 0].median()

# 获取除0以外每一列的最小值
nonzero_min_values = df[df != 0].min()

# 将结果存入文件进行输出
result_df = pd.DataFrame({
   'Max': max_values,
   'Min': min_values,
   'nonzero_min_values':nonzero_min_values,
   'Zero Count': zero_counts,
   'Null Count': null_counts,
   'Mean': mean_values,
   'Median': median_values,
   'Non-zero Mean': nonzero_mean_values,
   'Non-zero Median': nonzero_median_values
})

result_df.to_csv('Statistics.csv', index_label='Column Name')