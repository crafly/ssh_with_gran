import pandas as pd
import numpy as np
import jenkspy       #natural break
from jenkspy import JenksNaturalBreaks
import sys
import os

def str2array(str_para,sep=','):
    res=str_para.split(sep)
    res=[int(x) for x in res]
    return res

# GVF calculation function for determine number of cuts when using natural break
def calculate_gvf(data, nr_bins):
    data2=data*1
    total_variance = np.var(data2) * data2.shape[0]
    bins=jenkspy.jenks_breaks(data2, nr_bins)
    bins[0]=bins[0]-0.000001
    data2=pd.cut(data2, bins, labels=range(len(bins)-1))
    within_class_variance = sum(np.var(data [data2 == i])*(data[data2==i].shape[0]) for i in range(nr_bins))
    #within_class_variance = sum(np.var ( data [data2 == i] ) for i in
    #                            range(nr_bins))
    return 1 - (within_class_variance / total_variance)

if __name__ == "__main__":
    if len(sys.argv)<4:
        print("Usage: "+sys.argv[0]+" intput.csv 1,2,3 NR_categories natural")
        print("Possible methods are naturalbrk, equalfreq, equalwidth, mdlp.")
        sys.exit(0)
    # the data used
    data=np.loadtxt(sys.argv[1],delimiter=',',skiprows=1)
    # the columns to be discretized
    cols=str2array(sys.argv[2])
    deccol=data[:,-1]
    # the number of final values, discretized into NR_categories categories
    NR_categories=int(sys.argv[3])
    max_cate=NR_categories
    # the method used for discretization
    method=sys.argv[4]
    for i in range(len(cols)): # for each column do the discretizations
        # start discretization
        # at least python>3.10 is needed to support match
        match method:
            case "naturalbrk":
                bins=jenkspy.jenks_breaks(data[:,cols[i]], NR_categories)
            case "autonaturalbrk":
                # calculate the Goodness of Variance Fit (GVF) Threshold to
                # determine the optimal number of bins
                unique_values,counts=np.unique(data[:,cols[i]],
                                               return_counts=True)
                NR_categories=min(max_cate,len(unique_values))
                gvf_values=[]
                for b in range(2, NR_categories):
                    databk=data[:,cols[i]].copy()
                    gvf_values.append(calculate_gvf(databk,b))
                # Choose optimal number of bins based on GVF threshold (e.g., 0.9) using a loop
                optimal_bins = NR_categories  # Default to the maximum if no threshold is met
                for b in range(2,NR_categories):
                    gvf=gvf_values[b-2]
                    if gvf >= 0.8: 
                        optimal_bins = b
                        break
                NR_categories = optimal_bins
                #print(NR_categories)
                #exit(0)
                bins=jenkspy.jenks_breaks(data[:,cols[i]], NR_categories)
            case "equalfreq":
                res,bins=pd.qcut(data[:,cols[i]],NR_categories,
                                 labels=None, retbins=True,
                                 duplicates='drop')
            case "equalwidth":
                res,bins=pd.cut(data[:,cols[i]],NR_categories,
                                labels=range(NR_categories),
                                retbins=True)
            case default:
                print("Can not recognize your discretization method. \
                  Possible values are natura, equalfreq, equalwidth.")
                exit(0)
        # adjust the minimal value to avoid wrong discretization results
        bins[0]=bins[0]-0.000001
        data[:,cols[i]]=pd.cut(data[:,cols[i]],bins, labels=range(len(bins)-1))
    os.system("sed -n 1p "+sys.argv[1])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-1):
            print(data[i,j].astype(int), end=',')
        print(data[i,data.shape[1]-1].astype(float))
