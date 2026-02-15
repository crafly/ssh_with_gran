method="naturalbrk"
mkdir wuhanpic_$method
>wuhan_${method}_res.csv
>si.csv
for j in 7 8 9 10 12; do
  for ((i = 2; i <= 10; i++)); do
    echo "condition $j cuts $i" >>si.csv
    python ../discretization.py wuhanhouse.csv $j $i ${method} >wuhandisc.csv
    #qsta=$(python ../qsta_perm_granule.py wuhandisc.csv 13 ${j})
    #nmi=$(python ../NMI_granule.py wuhandisc.csv Continuous ${j} 13)
    #echo -e "attribute $j cuts $i, $qsta, $nmi" >>wuhan_${method}_res.csv
    shuf -n 1000 wuhandisc.csv >wuhandisc_sample.csv
    python ../DiffBetweenstrata.py wuhandisc_sample.csv $j 13 --ttype=Continuous >>si.csv
    #mv res.jpg wuhanpic_${method}/a${j}c${i}.jpg
  done
done
