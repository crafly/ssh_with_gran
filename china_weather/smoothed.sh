ds="china_weather.csv"
for stratum in 0 1 2; do
  python3 ../NMI_granule_laplace.py ${ds} Continuous ${stratum} 3
done
