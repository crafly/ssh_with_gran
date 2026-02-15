method=$1 # can be naturalbrk or equalfreq
rm -rf pic_$method
mkdir pic_$method
>si_${method}.csv
for ((j = 3; j <= 4; j++)); do
  #conditional feature
  for ((i = 2; i <= 10; i++)); do
    #cuts
    python ../discretization.py Cin_cont.csv $j $i ${method} >cindisc.csv
    # qsta=$(python ../qsta_perm_granule.py cindisc.csv 5 ${j})
    nmi=$(python ../NMI_granule.py cindisc.csv Nominal ${j} 5)
    echo -e "attribute $j # of strata $i, $nmi"
  done
done
for ((j = 3; j <= 4; j++)); do
  #conditional feature
  for ((i = 2; i <= 10; i++)); do
    python ../discretization.py Cin_cont.csv $j $i ${method} >cindisc.csv
    echo "condition $j # of strata $i," >>si_${method}.csv
    python ../DiffBetweenstrata.py cindisc.csv $j 5 --ttype=Discrete >>si_${method}.csv
    mv res.jpg pic_${method}/a${j}c${i}.jpg
  done
done
