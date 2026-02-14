ds="usa.csv"
shuf -n 201 ${ds} >${ds}_shuf.csv
for target in 5 8; do
  for stratum in 1 2 3; do
    python3 ../qsta_perm_granule.py ${ds} ${target} ${stratum} | tr -d "\n"
    echo -n ","
    python3 ../NMI_granule.py ${ds} Continuous ${stratum} ${target}
  done >res_${target}.csv
  cut -f 3,4 -d',' res_${target}.csv | sed "s/R_C=//g" | sed "s/H(s)=//g" | sed "s/,/ /g" >/tmp/points.csv
  python3 template.py
  mv a.pdf vi_${target}.pdf
done

# Uncomment to draw SI-graph
>si.csv
for target in 5 8; do
  for stratum in 1 2 3; do
    echo "Significance Matrix for $stratum" >>si.csv
    python3 ../DiffBetweenstrata.py ${ds} ${stratum} ${target} --ttype=Continuous >>si.csv
    mv res.jpg ${ds/\.csv/}_${target}_${stratum}.jpg
  done
done
