import numpy as np
import sys
import os
import pandas as pd
from sklearn import metrics
import math
from collections import Counter
from sklearn.metrics import mutual_info_score
import scipy
import scipy.stats
import random
np.random.seed(42)

klfunc="nofunc"

def cal_strata_entropy2(eqcls_d):
    #calculate the normalized entropy of the stratification granularity structure
    tol=0
    eqs=[]
    for i in range(len(eqcls_d)):
        tol+=len(eqcls_d[i])
    for i in range(len(eqcls_d)):
        eqs.append(1.0*len(eqcls_d[i])/tol)
    # add Laplace smoothing
    # 将概率分布转换为数组以便处理
    eqs_array = np.array(eqs)
    # 应用 Laplace 平滑: (count + 1) / (total + number_of_classes)
    smoothed_eqs = (eqs_array * tol + 1) / (tol + len(eqs_array))
    # 重新归一化以确保概率总和为1
    smoothed_eqs = smoothed_eqs / np.sum(smoothed_eqs)
    entropy=scipy.stats.entropy(smoothed_eqs,base=2)
    return entropy

def search_eqclass(matrix):
    """
    Construct a stratification using an attribute - optimized version
    """
    # 将矩阵的每一行转换为可哈希的元组，便于分组
    rows_as_tuples = [tuple(row) for row in matrix]
    
    # 使用字典来存储等价类，键为行的值，值为索引列表
    eqclass_dict = {}
    for idx, row_tuple in enumerate(rows_as_tuples):
        if row_tuple not in eqclass_dict:
            eqclass_dict[row_tuple] = []
        eqclass_dict[row_tuple].append(idx)
    
    # 提取等价类
    eqcls_end = list(eqclass_dict.values())
    
    # 构建 eqc_no 数组
    eqc_no = np.zeros(len(matrix))
    for c_no, indices in enumerate(eqcls_end):
        for obj in indices:
            eqc_no[obj] = c_no
            
    return eqcls_end, eqc_no

def generate_class_p(eqclass, objNum):
    #estimate the P(s=i) for each stratum    
    class_p = []
    for i in range(len(eqclass)):
        temp = len(eqclass[i]) / (objNum)
        class_p.append(temp)

    return class_p

def eqclass_d_list(eqclass, original_matrix, decatt):
    # calculate the target variable value set for each stratum
    objNum = len(original_matrix)
    decision_i = []
    classP_i = generate_class_p(eqclass, objNum)
    for si in eqclass:
        si_arry = np.array(si)
        si_decison = original_matrix[si_arry]
        i_decison = si_decison[:, decatt]
        decision_i.append(list(i_decison))
    return decision_i

def compute_kl_div_KDE(eqclass_decision, d_list, num_cuts):
    # 使用基于核密度估计(KDE)和自适应带宽计算相对熵, num_cuts is a fake variable
    eqcla_dec_ikl = []
    
    # 将整体数据转换为numpy数组并移除NaN值
    d_list_clean = np.array(d_list)[~np.isnan(d_list)]
    
    # 为整体分布构建KDE模型，使用自适应带宽选择
    if len(d_list_clean) > 1:
        # 使用Scott规则作为初始带宽估计
        scott_bw = len(d_list_clean) ** (-1/5)
        # 构建整体数据的KDE模型
        kde_global = scipy.stats.gaussian_kde(d_list_clean, bw_method=scott_bw)
    else:
        # 如果数据太少，则无法构建有意义的KDE
        # 为每个分层返回默认值或错误处理
        return [0.0] * len(eqclass_decision)
    
    # 确定评估点的范围（基于整体数据）
    min_val = np.min(d_list_clean)
    max_val = np.max(d_list_clean)
    # 扩展范围以覆盖更多区域
    range_ext = (max_val - min_val) * 0.1
    eval_points = np.linspace(min_val - range_ext, max_val + range_ext, 1000)
    
    # 计算全局PDF值
    global_pdf = kde_global(eval_points)
    # 避免零概率导致的数值问题
    global_pdf = np.maximum(global_pdf, 1e-10)
    # 归一化
    global_pdf = global_pdf / np.trapz(global_pdf, eval_points)
    
    # 对每个分层计算KL散度
    for i in range(len(eqclass_decision)):
        # 获取当前分层的数据并清理
        eqcls_data = np.array(eqclass_decision[i])
        eqcls_data_clean = eqcls_data[~np.isnan(eqcls_data)]
        
        if len(eqcls_data_clean) > 1:
            # 为当前分层构建KDE模型
            scott_bw_local = len(eqcls_data_clean) ** (-1/5)
            kde_local = scipy.stats.gaussian_kde(eqcls_data_clean, bw_method=scott_bw_local)
            
            # 计算局部PDF值
            local_pdf = kde_local(eval_points)
            # 避免零概率导致的数值问题
            local_pdf = np.maximum(local_pdf, 1e-10)
            # 归一化
            local_pdf = local_pdf / np.trapz(local_pdf, eval_points)
            
            # 计算KL散度: KL(P||Q) = ∫ P(x) * log(P(x)/Q(x)) dx
            # 其中P是局部PDF，Q是全局PDF
            kl_div = np.trapz(local_pdf * np.log(local_pdf / global_pdf), eval_points) / np.log(2)
            eqcla_dec_ikl.append(kl_div)
        else:
            # 数据不足时返回0
            eqcla_dec_ikl.append(0.0)
            
    return eqcla_dec_ikl

def compute_kl_div_smoothbin(eqclass_decision, d_list, num_cuts):
    # 使用基于平滑直方图计算相对熵
    eqcla_dec_ikl = []
    
    # 将整体数据转换为numpy数组并移除NaN值
    d_list_clean = np.array(d_list)[~np.isnan(d_list)]
    
    if len(d_list_clean) == 0:
        return [0.0] * len(eqclass_decision)
    
    # 确定全局数据的分箱边界
    _, bin_edges = np.histogram(d_list_clean, bins=num_cuts)
    
    # 计算全局数据的直方图分布
    global_hist, _ = np.histogram(d_list_clean, bins=bin_edges)
    
    # 平滑处理：添加小的常数避免零概率
    smoothing_factor = 1e-10
    global_hist_smooth = global_hist + smoothing_factor
    global_pdf = global_hist_smooth / np.sum(global_hist_smooth)
    
    # 对每个分层计算KL散度
    for i in range(len(eqclass_decision)):
        # 获取当前分层的数据并清理
        eqcls_data = np.array(eqclass_decision[i])
        eqcls_data_clean = eqcls_data[~np.isnan(eqcls_data)]
        
        if len(eqcls_data_clean) > 0:
            # 使用相同的分箱边界对当前分层数据进行分箱
            eqcls_hist, _ = np.histogram(eqcls_data_clean, bins=bin_edges)
            
            # 平滑处理：添加小的常数避免零概率
            eqcls_hist_smooth = eqcls_hist + smoothing_factor
            eqcls_pdf = eqcls_hist_smooth / np.sum(eqcls_hist_smooth)
            
            # 计算KL散度: KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
            # 其中P是局部PDF，Q是全局PDF
            kl_div = np.sum(eqcls_pdf * np.log(eqcls_pdf / global_pdf)) / np.log(2)
            eqcla_dec_ikl.append(kl_div)
        else:
            # 数据不足时返回0
            eqcla_dec_ikl.append(0.0)
            
    return eqcla_dec_ikl

def compute_kl_div_ML(eqclass_decision, d_list, num_cuts):
    # 使用基于正则化最大似然密度估计计算相对熵
    eqcla_dec_ikl = []
    
    # 将整体数据转换为numpy数组并移除NaN值
    d_list_clean = np.array(d_list)[~np.isnan(d_list)]
    
    if len(d_list_clean) == 0:
        return [0.0] * len(eqclass_decision)
    
    # 使用相同的方法对全局数据和各分层数据进行分箱
    # 确定全局数据的分箱边界
    global_hist, bin_edges = np.histogram(d_list_clean, bins=num_cuts)
    
    # 正则化最大似然估计：添加伪计数（拉普拉斯平滑）
    alpha = 1.0  # 正则化参数
    global_hist_reg = global_hist + alpha
    global_pdf = global_hist_reg / np.sum(global_hist_reg)
    
    # 对每个分层计算KL散度
    for i in range(len(eqclass_decision)):
        # 获取当前分层的数据并清理
        eqcls_data = np.array(eqclass_decision[i])
        eqcls_data_clean = eqcls_data[~np.isnan(eqcls_data)]
        
        if len(eqcls_data_clean) > 0:
            # 使用相同的分箱边界对当前分层数据进行分箱
            eqcls_hist, _ = np.histogram(eqcls_data_clean, bins=bin_edges)
            
            # 正则化最大似然估计：添加伪计数
            eqcls_hist_reg = eqcls_hist + alpha
            eqcls_pdf = eqcls_hist_reg / np.sum(eqcls_hist_reg)
            
            # 计算KL散度: KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
            # 其中P是局部PDF，Q是全局PDF
            kl_div = np.sum(eqcls_pdf * np.log(eqcls_pdf / global_pdf)) / np.log(2)
            eqcla_dec_ikl.append(kl_div)
        else:
            # 数据不足时返回0
            eqcla_dec_ikl.append(0.0)
            
    return eqcla_dec_ikl

def ICds_entropy(eqcla_dec_ikl, eqclass_p):
    # calculate the I_C(d,s) between the condition and target
    total_entropy = 0
    for i in range(len(eqclass_p)):
        temp_entropy = eqclass_p[i] * np.arctan(eqcla_dec_ikl[i]) * 2 / np.pi
        total_entropy = total_entropy + temp_entropy
    return total_entropy

def permutation_test(original_array,eqclass_list,permut_times, realval,dec_att_no,num_cuts):
    # permutation test for the continuous-valued conditional feature
    deci_list = original_array[:,dec_att_no]
    inorder_matrix = original_array.copy()
    NR_greater = 0
    eqclass_p = generate_class_p(eqclass=eqclass_list,objNum=len(deci_list))
    # Start permutation test
    for i in range(permut_times):
        np.random.shuffle(inorder_matrix[:,dec_att_no])
        eqcls_decision_list = eqclass_d_list(eqclass_list,inorder_matrix,dec_att_no)
        eqcla_dec_kldiv =klfunc(eqcls_decision_list,deci_list,num_cuts)
        temp = ICds_entropy(eqcla_dec_ikl=eqcla_dec_kldiv,eqclass_p=eqclass_p)
        if temp >= realval:
            NR_greater+=1
    return (1 + NR_greater) / (1+permut_times)

def permutation_test_cate(original_array,eqc_no,permut_times, entropy_d, realval,dec_att_no):
    # permutation test for the nominal conditional feature
    deci_list = original_array[:,dec_att_no]
    inorder_matrix = original_array.copy()
    NR_greater = 0
    # Start permutation test
    for i in range(permut_times):
        np.random.shuffle(inorder_matrix[:,dec_att_no])
        eqcls_decision_list = eqclass_d_list(eqclass_list,inorder_matrix,dec_att_no)
        MI=mutual_info_score(eqc_no,inorder_matrix[:,dec_att_no])/np.log(2)
        NMI=MI/entropy_d
        # NMI is the I_N(d,s)
        if NMI >= realval:
            NR_greater+=1
    return (1 + NR_greater) / (1+permut_times)

if __name__ == '__main__':
    if (len(sys.argv)<5):
        print(sys.argv[0]+" datafile.txt Continuous|Nominal Att_column Dec_column")
        print("Examples")
        print(sys.argv[0]+" catedata.txt Nominal 0 1")
        print(sys.argv[0]+" contdata.txt Continuous 0 1 kldivfunc")
        # 连续变量的情况下，最后一个参数是使用的kldiv方法，可以选择compute_kl_div_KDE, compute_kl_div_ML, compute_kl_div_smoothbin
        sys.exit(0)
    condatt=int(sys.argv[3])
    decatt=int(sys.argv[4])
    dec_type=sys.argv[2]
    #numerical_array=np.loadtxt(sys.argv[1],delimiter=',',skiprows=1)
    numerical_array=np.genfromtxt(sys.argv[1],delimiter=',',skip_header=1,missing_values='NA', filling_values=np.nan)
    
    # 使用np.isnan检查每个元素是否为np.nan，并使用.any(axis=1)找到包含np.nan的行
    mask = np.isnan(numerical_array).any(axis=1)
    # 取反mask以选择不包含np.nan的行
    numerical_array= numerical_array[~mask]
    decision_list = numerical_array[:, decatt]
    eqclass_list, eqc_no = search_eqclass(numerical_array[:, [condatt]])
    eqcls_d = eqclass_d_list(eqclass_list, numerical_array, decatt)
    strata_entropy=cal_strata_entropy2(eqcls_d)
    num_cuts=6
    if dec_type=='Continuous': # For continuous conditional features
        klfunc=globals().get(sys.argv[5])
        kl_div_i = klfunc(eqcls_d, decision_list,int(num_cuts))
        class_p = generate_class_p(eqclass_list, len(numerical_array))
        NMI=ICds_entropy(eqcla_dec_ikl=kl_div_i,eqclass_p=class_p)
        # here NMI is the I_C(d,s)
        p = permutation_test(numerical_array, eqclass_list, 1000, NMI,decatt,int(num_cuts))
        print('R_C=%.4f, H(s)=%.4f, B_C=%.4f, p-value=%.4f' %(NMI,strata_entropy,NMI/strata_entropy,p))
    else:                       # For Nominal conditional features
        mask = np.unique(decision_list)
        tmp = []
        for v in mask:
            tmp.append(np.sum(decision_list==v))
        # calculate the P(d) for each possible target variable value
        # add laplace smoothing
        pmf=np.array(tmp)+1
        pmf=pmf/decision_list.shape[0]
        entropy_dec=scipy.stats.entropy(pmf,base=2)       #H(d)
        MI=mutual_info_score(eqc_no,decision_list)/np.log(2) #I(d,s)
        NMI=MI/entropy_dec
        # here NMI is the I_N(d,s)
        p=permutation_test_cate(numerical_array, eqc_no, 1000, entropy_dec,NMI,decatt)
        print('R_N=%.4f, H(s)=%.4f, B_N=%.4f, p-value=%.4f' %(NMI,strata_entropy,NMI/strata_entropy,p))
