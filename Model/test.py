import numpy as np

# 创建一个示例矩阵
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 使用numpy的max和min函数计算每列的最大值和最小值
max_vals = np.max(matrix, axis=0)
min_vals = np.min(matrix, axis=0)

# 归一化每列的值到0到1之间
normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)

# 打印归一化后的矩阵
print(normalized_matrix)