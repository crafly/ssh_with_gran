ds="china_weather.csv"
for stratum in 0 1 2; do
  python3 ../qsta_perm_granule.py ${ds} 3 ${stratum} | tr -d "\n"
  echo -n ","
  python3 ../NMI_granule.py ${ds} Continuous ${stratum} 3
done >res.csv
cat res.csv
cut -f 3,4 -d',' res.csv | sed "s/R_C=//g" | sed "s/H(s)=//g" | sed "s/,/ /g" >/tmp/points.csv
python3 template.py
mv a.pdf vi.pdf
# Uncomment to draw SI-graph
>si.csv
for stratum in 0 1 2; do
  echo "Significance Matrix for $stratum" >>si.csv
  python3 ../DiffBetweenstrata.py ${ds} ${stratum} 3 --ttype=Continuous >>si.csv
  mv res.jpg ${ds/\.csv/}_${stratum}.jpg
done
