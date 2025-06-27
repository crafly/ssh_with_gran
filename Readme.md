I. File Lists:
fig1a.csv to fig1g.csv  #Corresponding to the illustrative dataset (a) to (e)
NMI_granule.py            #The program for calculating NMI based SSH and SSHG
qsta_perm_granule.py    #The program for calculating q-statistic based SSH and SSHG using permutation tests
DiffBetweenstrata.py # The program for testing the statistical differences between strata.

II. Steps for calculating the required measure:

1. prepare the data as an csv file separated by ',', and the first line includes the head.
2. When using the NMI_granule.py, direct execute the code in a terminal as follows:
   for nominal target variable
   \$ python3 NMI_granule.py catedata.txt Nominal 0 1
   for continuous target variable
   \$ python3 NMI_granule.py contdata.txt Continuous 0 1
   Here, the first parameter shows which data is used, the second parameter shows the type of conditional fatures, the third parameter is the column number corresponds to the conditional feature (the number starts from 0 rather than 1), the forth parameter is the column number corresponding to the target variable.
3. When using the qstatistics, direct execute the code in a terminal as follows:
   only suitable for continuous target variable
   \$ python3 qsta_perm_granule.py qstatistics data.csv 1 7
   Here the first parameter is fixed, the second parameter shows the data to be processed, the third and forth parameters shows the columns corresponding to the conditional feature and targer variable. This number starts from 0, also.
4. To justify the difference between strata:
   
   \$ python3 DiffBetweenstrata.py data.csv 0 1 --ttype=continuous    #for Continuous-valued conditions

   \$ python3 DiffBetweenstrata.py data.csv 0 1 --ttype=Discrete    #for Discrete conditions
