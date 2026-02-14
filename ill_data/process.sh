echo "q and UEP pvalues"
for ds in fig3*.csv; do
  echo -n "${ds/\.csv/}: "
  python3 ../qsta_perm_granule.py ${ds} 1 0 | tr -d "\n"
  echo -n ", "
  python3 ../NMI_granule.py ${ds} Nominal 0 1
done

for ds in fig3*.csv; do
  echo "SI-Graph pvalues in ${ds}"
  python3 ../DiffBetweenstrata.py ${ds} 0 1 --ttype=Discrete
  mv res.jpg ${ds}.jpg
done
