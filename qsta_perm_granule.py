import numpy as np
import random
import sys
import os
from sklearn.utils import resample
from scipy.special import comb
import scipy
import scipy.stats
np.random.seed(42)

NR_iter=1000
sig_level=0.05

def cal_strata_entropy(eqcls_d):
    #calculate the normalized entropy of the stratification granularity structure
    tol=0
    eqs=[]
    for i in range(len(eqcls_d)):
        tol+=len(eqcls_d[i])
    for i in range(len(eqcls_d)):
        eqs.append(1.0*len(eqcls_d[i])/tol)
    entropy=scipy.stats.entropy(eqs,base=2)
    #hmax=np.log(tol)
    #entropy=1.0-entropy/hmax
    return entropy

def qstatistics(SST, strata):
    #calculate the q-statistic
    SSW=0
    U_size=0
    for i in range(len(strata)):
        l_size=len(strata[i])
        U_size+=l_size
        SSW+=np.var(strata[i])*l_size
    return 1.0-SSW/SST

def str2array(str_para,sep=','):
    #convert a string separated by ',' to an array
    res=str_para.split(sep)
    res=[int(x) for x in res]
    return res

def shuffleequ(dec_eqc):
    # reshuffle the target variable
    flat_list= [n for a in dec_eqc for n in a ]
    NR_rec=len(flat_list)
    for i in range(NR_rec):
        j=random.randrange(0,NR_rec)
        tmp=flat_list[i]
        flat_list[i]=flat_list[j]
        flat_list[j]=tmp
    reslist=[]
    pos=0
    for i in range(len(dec_eqc)):
        a=[]
        for j in range(len(dec_eqc[i])):
            a.append(flat_list[pos])
            pos+=1
        reslist.append(a)
    return reslist

def determine_power(measure,data,condatts,decatts):
    # permutation test for one conditional feature
    cond_eql,eqc_no=search_eqclass(data[:,condatts])
    strata_entropy=cal_strata_entropy(cond_eql)
    SST=np.var(data[:,dec_atts])*data.shape[0]
    strata=[]
    for i in cond_eql:
        strata.append(data[i,decatts])
    indexV=measure(SST,strata)
    p=1.0
    # The permutation test starts here
    for i in range(NR_iter):
        local_strata=[]
        cond_eql=shuffleequ(cond_eql)
        for i in cond_eql:
            local_strata.append(data[i,decatts])
        l_indexV=measure(SST,local_strata)
        if(l_indexV>=indexV): p+=1.0
    p=1.0*p/1001.0
    return indexV,strata_entropy,indexV*strata_entropy,p

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

if __name__ == "__main__":
    if len(sys.argv)<4:
        print(sys.argv[0]+" data.csv 1 7")
        sys.exit(0)
    measure=globals().get("qstatistics")
    csvname=sys.argv[1]
    dec_atts=str2array(sys.argv[2]) # target variable
    attset1=str2array(sys.argv[3])  # conditional feature
    # read the data
    #data=np.loadtxt(csvname,delimiter=',',skiprows=1)
    data=np.genfromtxt(csvname,delimiter=',',skip_header=1,missing_values='NA', filling_values=np.nan)
    # 使用np.isnan检查每个元素是否为np.nan，并使用.any(axis=1)找到包含np.nan的行
    mask = np.isnan(data).any(axis=1)
    # 取反mask以选择不包含np.nan的行
    data= data[~mask]
    # calculate the q, G(s), SSHG, and the corresponding p-value
    index,strata_g,indexg,p=determine_power(measure, data.copy(), attset1, dec_atts)
    #print('q=%.4f, H(s)=%.4f, %.4f, p-value=%.4f' %(index, strata_g,index/strata_g,p))
    print('q=%.4f, p-value=%.4f' %(index, p))
