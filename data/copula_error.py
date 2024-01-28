import numpy as np
from scipy.stats import rankdata, entropy
import csv
import pandas as pd

#整段代码的目的就是计算两个离散变量的联合概率分布
def discrete_copula_entropy(data1, data2):
    joint_probs = np.zeros((len(set(data1)), len(set(data2))))
    for i, x in enumerate(set(data1)):
        for j, y in enumerate(set(data2)):
            joint_probs[i, j] = np.sum((data1 == x) & (data2 == y)) / len(data1)
            print("joint_probs[{}, {}]: {}".format(i, j, joint_probs[i, j]))

    #copula_entropy = -np.nansum(joint_probs * np.log2(joint_probs))
    copula_entropy = - np.nansum(np.where(joint_probs > 0, joint_probs * np.log2(joint_probs), 0))
    return copula_entropy

def calculate(csv_file):
    df = pd.read_csv(csv_file, encoding='GBK')
    column_count = df.shape[1]
    copula_entropy_matrix = np.zeros((2, column_count))
    result = df['Value']
    for j, column_name in enumerate(df.columns[0:-1]):
        copula_entropy = discrete_copula_entropy(df.iloc[1:, j], result)
        copula_entropy_matrix[0, j] = column_name
        copula_entropy_matrix[1, j] = copula_entropy
    print("Copula熵:", copula_entropy_matrix)

    output_df = pd.DataFrame(copula_entropy_matrix.T, columns=['Column Name', 'Copula Entropy'])
    output_df.to_csv('copula_output.csv', index=False)
    print("Copula熵已保存到 copula_output.csv 中")

    return copula_entropy_matrix

np.set_printoptions(threshold=np.inf)

# 传入一组 CSV 文件路径
file_path = "output_fill.csv"
copula_entropy = calculate(file_path)
#print("Copula熵:", copula_entropy)