import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# 1. 加载CSV文件
df = pd.read_csv('boruta_selected_features.csv')
data = pd.DataFrame(df)

# 2. 将数据分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.1, random_state=1)

# 3. 使用等频标量算法进行离散化处理
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
discretized_train_data = discretizer.fit_transform(train_data)
discretized_test_data = discretizer.transform(test_data)

# 4. 利用RS理论获取约简区域
def rs_theory(data):
    n = data.shape[1] - 1  # 属性的个数
    reduct_region = set(range(n))  # 初始化约简区域为所有属性

    for i in range(n):  # 遍历每个属性
        candidate = reduct_region.copy()
        candidate.remove(i)

        if dependency(data, candidate, reduct_region - candidate):
            reduct_region = candidate

    return reduct_region

# 如果is_dependency为True，则表示所有分组中的选民决策相同，不存在依赖关系。
# 如果is_dependency为False，则表示至少存在一个分组中的选民决策不同，存在依赖关系。(示例)
def dependency(data, condition, decision):
    data = pd.DataFrame(data)
    data_groups = data.groupby(list(condition))
    is_dependency = True

    for name, group in data_groups:
        #if len(group[decision].unique()) > 1:
        if len(group[list(decision)].drop_duplicates()) > 1:
            is_dependency = False
            break

    return is_dependency

reduct_region = rs_theory(discretized_train_data)

# 5. 遗传算法计算属性重要性
def calculate_importance(data, reduct_region, population_size, generations, crossover_rate, mutation_rate, inversion_rate):
    n_attributes = data.shape[1] - 1  # 属性的个数
    n_instances = data.shape[0]  # 实例的个数
    population = np.random.randint(2, size=(population_size, n_attributes))  # 随机初始化种群
    fitness_scores = np.zeros(population_size)  # 适应度函数值

    # 构建映射模式
    mapping_pattern = np.zeros(n_attributes, dtype=bool)
    mapping_pattern[list(reduct_region)] = True

    for generation in range(generations):
        # 计算适应度函数值
        for i in range(population_size):
            # 计算缩减区域C
            candidate_condition = np.where(population[i] == 0)[0]
            candidate_mapping_pattern = mapping_pattern.copy()
            candidate_mapping_pattern[candidate_condition] = False

            gamma_C = np.sum(np.abs(data[:, :-1] - data[:, -1].reshape(-1, 1))[candidate_mapping_pattern])

            # 计算缩减区域C-a
            gamma_C_a = np.sum(np.abs(data[:, :-1] - data[:, -1].reshape(-1, 1))[mapping_pattern])

            # 计算属性的重要性
            importance = (gamma_C - gamma_C_a) / gamma_C

            # 计算属性子集的约简度
            reduction_degree = np.sum(np.abs(candidate_mapping_pattern - mapping_pattern)) / len(data)

            # 更新适应度函数值
            fitness_scores[i] = reduction_degree

        # 选择操作
        parents = np.random.choice(np.arange(population_size), size=int(population_size / 2), replace=False, p=fitness_scores / np.sum(fitness_scores))
        offspring = []

        # 交叉操作
        for i in range(0, len(parents), 2):
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, n_attributes - 1)
                offspring.append(np.concatenate((population[parents[i], :crossover_point], population[parents[i+1], crossover_point:])))
                offspring.append(np.concatenate((population[parents[i+1], :crossover_point], population[parents[i], crossover_point:])))
            else:
                offspring.append(population[parents[i]])
                offspring.append(population[parents[i+1]])

        offspring = np.array(offspring)

        # 突变操作
        for i in range(offspring.shape[0]):
            for j in range(offspring.shape[1]):
                if np.random.rand() < mutation_rate:
                    offspring[i, j] = 1 - offspring[i, j]

        # 反演操作
        for i in range(offspring.shape[0]):
            if np.random.rand() < inversion_rate:
                inversion_point = np.random.randint(n_attributes)
                offspring[i, :inversion_point] = 1 - offspring[i, :inversion_point]

        # 更新种群
        population[:offspring.shape[0]] = offspring

    # 返回最优属性重要性值
    best_individual = population[np.argmax(fitness_scores)]
    best_mapping_pattern = mapping_pattern.copy()
    best_mapping_pattern[best_individual == 1] = True

    # 构建最优映射模式
    best_diff_count = np.sum(np.abs(data[:, :-1] - data[:, -1].reshape(-1, 1))[best_mapping_pattern], axis=0)
    best_importance = (np.sum(best_diff_count) / n_instances) / len(reduct_region)

    return best_importance, best_mapping_pattern

# 调用函数计算属性重要性和构建映射模式
population_size = 70
generations = 250
crossover_rate = 0.3
mutation_rate = 0.05
inversion_rate = 0.05

attribute_importance, mapping_pattern = calculate_importance(discretized_train_data, reduct_region, population_size, generations, crossover_rate, mutation_rate, inversion_rate)

print("属性重要性:", attribute_importance)
print("映射模式:", mapping_pattern)