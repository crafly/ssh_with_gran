import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import networkx as nx
import argparse
import matplotlib.pyplot as plt
from collections import Counter
#from statsmodels.stats.contingency_tables import Table
import itertools

from math import factorial

def compute_table_probability(table, row_sums, col_sums, total):
    """
    计算给定表格的概率。
    :param table: 当前表格 (二维列表)
    :param row_sums: 每行的总和 (一维列表)
    :param col_sums: 每列的总和 (一维列表)
    :param total: 总样本数
    :return: 表格的概率
    """
    numerator = 1
    for r in row_sums:
        numerator *= factorial(r)
    for c in col_sums:
        numerator *= factorial(c)

    denominator = factorial(total)
    for row in table:
        for cell in row:
            denominator *= factorial(cell)

    return numerator / denominator


def enumerate_all_tables(row_sums, col_sums):
    """
    枚举所有满足边际总和约束的表格。
    :param row_sums: 每行的总和 (一维列表)
    :param col_sums: 每列的总和 (一维列表)
    :return: 所有可能的表格 (生成器)
    """
    def generate(row_sums, col_sums):
        if len(row_sums) == 1:  # 最后一行直接由列和决定
            yield [col_sums[:]]
        else:
            first_row = []
            for comb in itertools.combinations_with_replacement(range(len(col_sums)), row_sums[0]):
                counts = [comb.count(i) for i in range(len(col_sums))]
                if all(c >= 0 for c in [col_sums[j] - counts[j] for j in range(len(col_sums))]):
                    for sub_table in generate(row_sums[1:], [col_sums[j] - counts[j] for j in range(len(col_sums))]):
                        yield [counts] + sub_table

    return generate(row_sums, col_sums)


def fisher_exact_test(data):
    """
    多类别 Fisher 精确检验。
    :param data: 列联表 (二维列表)
    :return: p 值
    """
    # 计算行和、列和和总样本数
    row_sums = [sum(row) for row in data]
    col_sums = [sum(col) for col in zip(*data)]
    total = sum(row_sums)

    # 观察到的表格概率
    observed_prob = compute_table_probability(data, row_sums, col_sums, total)

    # 枚举所有可能的表格并计算 p 值
    p_value = 0
    for table in enumerate_all_tables(row_sums, col_sums):
        table_prob = compute_table_probability(table, row_sums, col_sums, total)
        if table_prob <= observed_prob:  # 累加更极端的表格
            p_value += table_prob

    return p_value


# 定义超几何分布的概率计算函数
def hypergeometric_probability(table, row_totals, col_totals):
    """
    计算给定列联表的概率（基于超几何分布）
    :param table: 当前列联表 (二维数组)
    :param row_totals: 每行的边际总和
    :param col_totals: 每列的边际总和
    :return: 概率值
    """
    # 分母部分
    denominator = factorial(sum(row_totals))
    for total in col_totals:
        denominator *= factorial(total)
    
    # 分子部分
    numerator = 1
    for total in row_totals:
        numerator *= factorial(total)
    for row in table:
        for cell in row:
            numerator *= factorial(cell)
    
    return numerator / denominator

# 随机生成满足边际总和条件的列联表
def generate_random_table(row_totals, col_totals):
    """
    使用随机方法生成一个满足边际总和条件的列联表
    :param row_totals: 每行的边际总和
    :param col_totals: 每列的边际总和
    :return: 随机生成的列联表
    """
    table = np.zeros((len(row_totals), len(col_totals)), dtype=int)
    remaining_col_totals = col_totals.copy()
    
    for i in range(len(row_totals)):
        row_total = row_totals[i]
        for j in range(len(col_totals) - 1):
            if row_total == 0 or remaining_col_totals[j] == 0:
                table[i][j] = 0
            else:
                max_val = min(row_total, remaining_col_totals[j])
                value = np.random.randint(0, max_val + 1)
                table[i][j] = value
                row_total -= value
                remaining_col_totals[j] -= value
        table[i][-1] = row_total  # 最后一列填充剩余值
        remaining_col_totals[-1] -= row_total
    
    return table

# Fisher精确检验（蒙特卡洛模拟）
def fisher_exact_test2(observed_table, num_simulations=1000):
    """
    使用蒙特卡洛模拟进行多分类变量的Fisher精确检验
    :param observed_table: 观察到的列联表 (二维数组)
    :param num_simulations: 蒙特卡洛模拟的次数
    :return: p值
    """
    observed_table = np.array(observed_table)
    row_totals = observed_table.sum(axis=1)  # 每行的边际总和
    col_totals = observed_table.sum(axis=0)  # 每列的边际总和
    
    # 计算观察表格的概率
    observed_prob = hypergeometric_probability(observed_table, row_totals, col_totals)
    
    # 蒙特卡洛模拟
    extreme_count = 0
    for _ in range(num_simulations):
        random_table = generate_random_table(row_totals, col_totals)
        random_prob = hypergeometric_probability(random_table, row_totals, col_totals)
        if random_prob <= observed_prob:
            extreme_count += 1
    
    # 计算p值
    p_value = (extreme_count + 1) / (num_simulations + 1)  # 加1平滑处理
    return p_value

def fishertest(var_A,var_B):
    # 确定总类别数
    max_category = max(max(var_A), max(var_B))
    num_categories = int(max_category) + 1  # 类别从 0 开始编码
    
    # 构建列联表2Xnum_categories
    contingency_table = np.zeros((2, num_categories), dtype=int)

    # 计算频数
    for a in var_A:
        contingency_table[0, int(a)] += 1
    for b in var_B:
        contingency_table[1, int(b)] += 1
    # 使用 statsmodels 的 Table 类进行 Fisher 精确检验
    p_value = fisher_exact_test(contingency_table)
    # 下面是卡方检验
    #table = Table(contingency_table)
    #result = table.test_nominal_association()
    #p_value=result.pvalue
    return 1, p_value

def draw_graph_from_adjacency_matrix(matrix):
    """
    根据邻接矩阵绘制无向图。

    参数:
        matrix (list of list of int): 邻接矩阵。
    """
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加边到图中
    G.add_nodes_from(range(len(matrix)))
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix[i])):
            if matrix[i][j] == 1:
                G.add_edge(i, j)
    # 绘制图
    pos = nx.circular_layout(G)  # 定义布局
    plt.figure(figsize=(6, 6)) 
    nx.draw(G, pos, with_labels=True, node_color='lightblue',edgecolors='black',
            edge_color='black', node_size=3000, font_size=32) 
    plt.title("")
    axis = plt.gca()
    axis.set_xlim([1.1*x for x in axis.get_xlim()])
    axis.set_ylim([1.1*y for y in axis.get_ylim()])
    plt.tight_layout()
    plt.savefig("res.jpg", format='jpg',dpi=300)

def find_connected_components(distinguish_matrix):
    """
    使用深度优先搜索(DFS)找到所有连通分量。

    参数:
        distinguish_matrix (list of list of int): 区分矩阵，0表示不能区分，1表示能区分。

    返回:
        list of set: 每个集合包含可以合并的一组类别的索引。
    """
    n = len(distinguish_matrix)
    visited = [False] * n
    components = []

    def dfs(node, component):
        """
        深度优先搜索辅助函数。

        参数:
            node (int): 当前访问的节点。
            component (set): 当前正在构建的连通分量。
        """
        visited[node] = True
        component.add(node)
        for neighbor in range(n):
            if distinguish_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(n):
        if not visited[i]:
            component = set()
            dfs(i, component)
            components.append(component)

    return components

def discretize_target(target_series, bins=10):
    """
    将连续的目标变量离散化为若干区间。
    :param target_series: 目标变量序列
    :param bins: 区间数量
    :return: 离散化后的zhi
    """
    result,cuts=pd.cut(target_series,bins=bins,labels=range(bins),retbins=True)
    result=result.to_numpy()
    return result, list(cuts)

def discretize(target_series, cuts):
    """
    将连续的目标变量离散化为若干区间。
    :param target_series: 目标变量序列
    :param cuts: 断点
    :return: 离散化后的zhi
    """
    result=pd.cut(target_series,cuts,labels=range(len(cuts)-1))
    result=result.to_numpy()
    return result

def pairwise_test(df, condition_col_index, target_col_index=-1, bins=10,
                    target_type="Continuous"):
    """
    :param df: 数据集(DataFrame)
    :param condition_col_index: 条件变量的列索引
    :param target_col_index: 目标变量的列索引（默认为最后一列）
    :param bins: 离散化的区间数量
    :return: 每对条件变量值的p值字典
    """

    unique_values = df.iloc[:, condition_col_index].unique()
    unique_values = np.sort(unique_values)
    results = {}
    for i in unique_values:
        subset = df[df.iloc[:, condition_col_index] == i].iloc[:, target_col_index]
        print(subset.shape[0],end=",")
        for j in unique_values:
            value1 = i
            value2 = j
            if target_type=='Discrete':
                #results[(value1,value2)]=fishertest(
                results[(value1,value2)]=fishertest(
                        df[df.iloc[:,condition_col_index]==
                           value1].iloc[:, target_col_index],
                        df[df.iloc[:,condition_col_index]==
                           value2].iloc[:, target_col_index])
            else:
                total,cuts = discretize_target(df.iloc[:, target_col_index], bins=bins)
                local1=discretize(df[df.iloc[:, condition_col_index] == value1].iloc[:, target_col_index], cuts)
                local2=discretize(df[df.iloc[:, condition_col_index] == value2].iloc[:, target_col_index], cuts)
                results[(value1,value2)]=ks_2samp(local1,local2 )
    return results

def main():
    parser = argparse.ArgumentParser(description="使用排列检验测试不同条件变量值之间的目标变量分布是否统计显著")
    parser.add_argument("csv_file", type=str, help="CSV文件路径")
    parser.add_argument("condition_col_index", type=int, help="条件变量的列索引（从0开始计数）")
    parser.add_argument("tindex", type=int, default=-1,help="目标变量的列索引")
    parser.add_argument("--bins", type=int, default=5, help="离散化的区间数量")
    parser.add_argument("--ttype", type=str, default="Continuous",
                        help="Continuous or Discrete")
    args = parser.parse_args()
    # 加载数据
    df = pd.read_csv(args.csv_file)
    
    # 执行排列检验
    try:
        results = pairwise_test(df, args.condition_col_index, args.tindex, bins=args.bins,
                                  target_type=args.ttype)
        unique_values = df.iloc[:, args.condition_col_index].unique()
        size=np.max(unique_values)+1
        res=np.zeros((size,size))
        # 打印结果
        for (value1, value2), result in results.items():
            res[value1,value2]=result[1]<0.01
            #res[value2,value1]=result[1]<0.01
        for (value1, value2), result in results.items():
            if (res[value1,value2]!=res[value2,value1]):
                # contradiction, set it to non-discernable
                res[value1,value2]=0
                res[value2,value1]=0
            if (value1==value2):
                res[value1,value2]=1
        draw_graph_from_adjacency_matrix(1-res)
        groups=find_connected_components(1-res)
        print(1.0*np.sum(1-res)/size/size,len(groups),"")
        #for idx, group in enumerate(groups):
        #    print(f"Group {idx + 1}: {group}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
