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

def cal_strata_entropy2(eqcls_d):
    #calculate the normalized entropy of the stratification granularity structure
    tol=0
    eqs=[]
    for i in range(len(eqcls_d)):
        tol+=len(eqcls_d[i])
    for i in range(len(eqcls_d)):
        eqs.append(1.0*len(eqcls_d[i])/tol)
    entropy=scipy.stats.entropy(eqs,base=2)
    #hmax=np.log2(tol)
    #entropy=1.0-entropy/hmax
    return entropy

def cal_strata_entropy(eqcls_d):
    #calculate the normalized entropy of the stratification granularity structure
    tol=0
    eqs=[]
    for i in range(len(eqcls_d)):
        tol+=len(eqcls_d[i])
    for i in range(len(eqcls_d)):
        eqs.append(1.0*len(eqcls_d[i])/tol)
    entropy=scipy.stats.entropy(eqs)
    #hmax=np.log2(tol)
    #entropy=1.0-entropy/hmax
    return entropy

def search_eqclass(matrix):
    #Construct a stratifucation using an attribute
    eqclass = []
    for i in range(len(matrix)):
        sublist = []
        sublist.append(i)
        for j in range(len(matrix)):
            # whether all the values are equal
            a = matrix[i]
            b = matrix[j]
            c = (a == b)
            if (c.all()) and i != j:
                sublist.append(j)
        eqclass.append(sublist)
    for item in eqclass:
        item.sort()
    demon = list(set(tuple(t) for t in eqclass))
    eqcls_end = [list(v) for v in demon]
    eqc_no=np.zeros((len(matrix)))
    for c_no in range(len(eqcls_end)):
        for obj in eqcls_end[c_no]:
            eqc_no[obj]=c_no
    return eqcls_end,eqc_no

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

def compute_kl_div(eqclass_decision, d_list, num_cuts):
    # compute the relative entropy between $f_i$ and $f$ for each stratum
    eqcla_dec_ikl = []
    k = num_cuts
    decision_list_div, bins = pd.cut(d_list, k, labels=range(k), retbins=True)
    decision_list_div_p = np.bincount(decision_list_div.codes)
    decision_list_div_p = decision_list_div_p / decision_list_div_p.sum()
    for i in range(len(eqclass_decision)):
        # for $f_i$
        eqcls_decision_i = eqclass_decision[i]
        eqcls_decision_i_div = pd.cut(eqcls_decision_i, bins, labels=range(k))
        eqcls_decision_i_div_p = np.bincount(eqcls_decision_i_div.codes)
        eqcls_decision_i_div_p = eqcls_decision_i_div_p / eqcls_decision_i_div_p.sum()
        if eqcls_decision_i_div_p.shape[0] < k:
            eqcls_decision_i_div_p = list(eqcls_decision_i_div_p)
            for i in range(len(eqcls_decision_i_div_p), k):
                eqcls_decision_i_div_p.append(0)
            eqcls_decision_i_div_p = np.array(eqcls_decision_i_div_p)
        i_kl_div = scipy.special.kl_div(eqcls_decision_i_div_p, decision_list_div_p).sum()/np.log(2)
        eqcla_dec_ikl.append(i_kl_div)
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
        eqcla_dec_kldiv =compute_kl_div(eqcls_decision_list,deci_list,num_cuts)
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
        print(sys.argv[0]+" contdata.txt Continuous 0 1")
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
    eqclass_list, eqc_no = search_eqclass(numerical_array[:, condatt])
    eqcls_d = eqclass_d_list(eqclass_list, numerical_array, decatt)
    strata_entropy=cal_strata_entropy2(eqcls_d)
    num_cuts=6
    if dec_type=='Continuous': # For continuous conditional features
        kl_div_i = compute_kl_div(eqcls_d, decision_list,int(num_cuts))
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
        pmf=np.array(tmp)/decision_list.shape[0]
        entropy_dec=scipy.stats.entropy(pmf,base=2)       #H(d)
        MI=mutual_info_score(eqc_no,decision_list)/np.log(2) #I(d,s)
        NMI=MI/entropy_dec
        # here NMI is the I_N(d,s)
        p=permutation_test_cate(numerical_array, eqc_no, 1000, entropy_dec,NMI,decatt)
        print('R_N=%.4f, H(s)=%.4f, B_N=%.4f, p-value=%.4f' %(NMI,strata_entropy,NMI/strata_entropy,p))
