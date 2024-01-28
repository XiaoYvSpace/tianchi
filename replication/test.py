#现在已经有了条件属性对决策属性的缩减区域，请利用计算属性等式IM_\left ( C,D \right ) \left ( a \right ) =\frac{\gamma _{C}(D)-\gamma_{C-a} (D)}{\gamma _{C}(D)}  和适应度函数\tau _{C}=\frac{ {\textstyle \sum_{i=1}^{m}}  \underline{\left | (C-D_{i}) \right |} }{\left | U \right | }  ,用遗传算法计算属性的重要性。其中，C表示条件属性子集，D表示决策属性子集，U表示所有对象的非空有限集。给出相应的python代码

import random
import pandas as pd

# 1. 加载CSV文件
df = pd.read_csv('boruta_selected_features.csv')

def calculate_importance(C, D, U):
    gamma_c = len(C.intersection(D)) / len(D)  # 计算 gamma_C(D)
    importance = []

    for attribute in C:
        C_a = C.difference({attribute})  # 构建 C-a
        gamma_c_a = len(C_a.intersection(D)) / len(D)  # 计算 gamma_C-a(D)
        im = (gamma_c - gamma_c_a) / gamma_c  # 计算 IM(C, D)(a)
        importance.append(im)

    return importance

def fitness_function(C, D, U):
    total_diff = sum(len(C.difference({d})) for d in D)  # 计算属性差异的累加和
    return total_diff / len(U)  # 返回适应度值

def genetic_algorithm(C, D, U, population_size, generations):
    population = []

    for _ in range(population_size):
        individual = random.sample(C, k=random.randint(1, len(C)))  # 随机生成个体
        population.append(individual)

    for generation in range(generations):
        fitness_scores = []

        for individual in population:
            fitness_scores.append(fitness_function(set(individual), D, U))  # 计算每个个体的适应度值

        # 选择和繁殖操作...

    # 返回最优个体或属性重要性

# 测试代码
# C = {'attr1', 'attr2', 'attr3', 'attr4'}
# D = {'decision1', 'decision2'}
# U = {'obj1', 'obj2', 'obj3', 'obj4', 'obj5'}

# 将除了最后一列的所有列设置为C（条件属性）
C = set(df.columns[:-1])

# 将最后一列设置为D（决策属性）
D = set(df.columns[-1])

# 创建U（所有对象的集合）
U = set(df[df.columns[-1]])

# 使用C、D和U进行后续的属性重要性计算或遗传算法操作
# ...

# 示例：打印C、D和U的值以进行验证
print("C: ", C)
print("D: ", D)
print("U: ", U)

importance = calculate_importance(C, D, U)
print(importance)

genetic_algorithm(C, D, U, population_size=50, generations=100)