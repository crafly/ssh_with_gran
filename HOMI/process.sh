method=$1 #naturalbrk or equalfreq
rm -rf pic_$method
mkdir pic_$method
>si_${method}.csv
>res_${method}.log
for j in 9 11; do
  #conditional feature
  for ((i = 2; i <= 10; i++)); do
    echo "condition $j cuts $i," >>si_${method}.csv
    #cuts
    python ../discretization.py south_origin.csv $j $i ${method} >southdisc.csv
    qsta=$(python ../qsta_perm_granule.py southdisc.csv 12 ${j})
    nmi=$(python ../NMI_granule.py southdisc.csv Continuous ${j} 12)
    echo -e "attribute $j cuts $i, $qsta, $nmi" >>res_${method}.log
    python ../DiffBetweenstrata.py southdisc.csv $j 12 --ttype=Continuous >>si_${method}.csv
    mv res.jpg pic_${method}/a${j}c${i}.jpg
  done
  grep "attribute ${j}" res_${method}.log | cut -f 4,5 -d',' | sed "s/R_C=//g" | sed "s/H(s)=//g" | sed "s/,/ /g" >/tmp/points.csv
  python3 template.py
  mv /tmp/a.pdf ${method}_vi_${j}_${i}.pdf
  mv a.tif ${method}_vi_${j}_${i}.tif
done
